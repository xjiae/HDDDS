import torch
from torch.utils.data import DataLoader, Subset
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from hddds import *
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from utils import *
import warnings
warnings.filterwarnings("ignore")
import gc
from tqdm.notebook import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
from models.lstm import StackedLSTM
from utils import *
from explainers.intgrad import IntGradExplainer
from explainers.explainer import TopKR2B
from models import *
from sklearn.model_selection import train_test_split
import pynvml
from pynvml.smi import nvidia_smi
from my_openxai import Explainer
pynvml.nvmlInit()
torch.manual_seed(1234)

def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()
    best = {"loss": sys.float_info.max}
    loss_history = []
    model = model.double()
    for e in tqdm(range(n_epochs)):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            x = batch["x"].cuda()
            y = batch["y"].cuda()
            ww = batch['exp'].cuda()
            y_pred = model(x, ww)
            loss = loss_fn(y_pred.double(), y.double())
            loss.backward()           
            epoch_loss += loss.item()
            optimizer.step()
        
        loss_history.append(epoch_loss)
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1
        # break
        
    return best, loss_history


def inference(dataset, model, batch_size, exp):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    k = 10
    if exp == "VG":
            # explainer = VGradExplainer(is_batched=True, r2b_method=TopKR2B(model.in_shape, k))
        explainer = Explainer(method="grad", model=XWrapper(model, x_to_explain))
    elif exp == 'IG':
        # explainer = Explainer(method="ig", model=XWrapper(model, x_to_explain), dataset_tensor=w_train)
        explainer = IntGradExplainer(is_batched=True, num_steps=20, r2b_method=TopKR2B(model.in_shape, k))
    model = nn.DataParallel(model)
    y_true, y_pred, w_true, w_pred = [], [], [], []
    # cnt = 0
    for batch in tqdm(dataloader):
        # cnt += 1
        # if cnt == 3:
            # breakpoint()
        x = batch["x"].cuda()
        y = batch["y"].cuda()
      
        x_to_explain = x[0]
        # w_train = torch.stack([torch.zeros_like(x_to_explain)] * 100)
        # x_train = torch.stack([torch.rand(x_to_explain.shape)] * 100)
        x_train = x_to_explain + torch.rand(100, *x_to_explain.shape).cuda()
        # x_train = x_train
        
        # x_train = torch.rand(x.shape)
       
        if exp == "LIME":
            explainer = Explainer(method="lime", model=model, dataset_tensor=x_train)
        elif exp == " SHAP":
            explainer = Explainer(method="shap", model=model, dataset_tensor=x_train)
            
        
        
        # torch nn.LSTM doesn't allow backward in eval mode, so we make it train briefly
        model.train()   
        # lbl_test = torch.randint(0,10, (batch_size,))  
        # w_test = torch.ones(x.shape)    
        if "G" in exp:
            w_p = explainer.get_explanation(model, x)  
        else:
            # x_to_explain = x_to_explain.cuda()
            y_test = y.view(1,1)
            w_p = explainer.get_explanation(x_to_explain.unsqueeze(0), y_test)  
        model.eval()
        y_p = model(x)

        y_true.append(y.cpu())
        if "G" in exp:
            y_pred.append(torch.max(y_p.detach().cpu(), dim=1)[0])
        else:
            y_pred.append(torch.max(y_p.detach().cpu()).view(1,1))
        w_true.append(batch["exp"].cpu().view(w_p.shape))
        w_pred.append(w_p.detach().cpu())   
    return (
        torch.cat(y_true),
        torch.cat(y_pred),
        torch.cat(w_true),
        torch.cat(w_pred)
    )   
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", type=str, default='hai_sliding_100')
    parser.add_argument("-model", type=str, default='LSTM')
    parser.add_argument("-exp", type=str, default='IG')
    parser.add_argument("-batch_size_train", type=int, default=20)
    parser.add_argument("-batch_size_test", type=int, default=200)
    parser.add_argument("-epoch", type=int, default=10)
    parser.add_argument("-test_only", type=int, default=0)
    parser.add_argument("-seed", type=int, default=1234)
    
    
    
    args, unknown = parser.parse_known_args()
    return args

def main(args):

        
    ds_name = args.ds
    ds_name_train = ds_name + "_train"
    ds_name_test = ds_name + "_test"
    model_name = args.model
    explainer = args.exp
    sample = False
    
    test_dataset = get_dataset(ds_name_test)
    d = test_dataset.num_features
    # too expensive to compute the whole dataset
    if explainer == "LIME" or explainer == "SHAP":
        _, indices = train_test_split(range(len(test_dataset)), test_size=200, stratify=test_dataset.y, random_state=args.seed)
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        # breakpoint()


    
    if args.test_only == 0:
    
        train_dataset = get_dataset(ds_name_train)

        model = SimpleLSTM(in_shape = (d,), out_shape = (1,)).double()

        model = nn.DataParallel(model)
        
        model.cuda()
        model.train()
        BEST_MODEL, LOSS_HISTORY = train(train_dataset, model, args.batch_size_train, args.epoch) 
        

        with open(f"models/LSTM/SimpleLSTM_{ds_name}.pt", "wb") as f:
            torch.save(
                {
                    "state": BEST_MODEL["state"],
                    "best_epoch": BEST_MODEL["epoch"],
                    "loss_history": LOSS_HISTORY,
                },
                f,
            )

    with open(f"models/LSTM/SimpleLSTM_{ds_name}.pt", "rb") as f:
    # with open(f"models/LSTM/SimpleLSTM.pt", "rb") as f:
        SAVED_MODEL = torch.load(f)


    loaded_model = SimpleLSTM(in_shape = (d,), out_shape = (1,)).double()
    
    new = {k.replace('module.',''):v for k,v in SAVED_MODEL["state"].items()}
    loaded_model.load_state_dict(new)
    
    
    loaded_model.eval().cuda()
    y_true, y_pred, w_true, w_pred = inference(test_dataset, loaded_model, args.batch_size_test, args.exp) 
    
    acc, f1, fpr, fnr = summary(y_true.cpu().numpy(), y_pred.cpu().numpy(), score = True)
    f = open(f"results/{ds_name}_experiment_result_model.txt", "a")
    f.write(f"{model_name} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\ \n")
    f.close()
    acc, f1, fpr, fnr = summary(w_true.cpu().numpy().flatten(), w_pred.cpu().numpy().flatten(), score = True)
    f = open(f"results/{ds_name}_experiment_result_explainer.txt", "a")
    f.write(f"{model_name} | {explainer} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\ \n")
    f.close()
            

if __name__ == '__main__':
    args = parse_args()
    # while True:
    #     usage = nvidia_smi.getInstance().DeviceQuery('memory.used')['gpu'][0]['fb_memory_usage']['used']
    #     if usage < 100:
    #         break
    main(args) 
    breakpoint()
    
    
        

    