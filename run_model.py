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
# warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
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
from explainers.intgrad import PlainGradExplainer, IntGradExplainer
from explainers import *
from models import *
N_HIDDENS = 200
N_LAYERS = 3
BATCH_SIZE = 50
epoch = 10
num_features = 86

def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()
    # epochs = trange(n_epochs, desc="training")
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
            
            # alpha = torch.ones_like(x)
            
  
            y_pred = model(x, ww)
            loss = loss_fn(y_pred.double(), y.double())

            loss.backward()
            
            epoch_loss += loss.item()
            optimizer.step()
            # breakpoint()
            # break
        
        loss_history.append(epoch_loss)
        # epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1
        # break
        
    return best, loss_history

def anomaly_detection(anomaly_score, threshold, total_ts):
    b, a = signal.butter(1, 0.02, btype='lowpass')
    distance = signal.filtfilt(b,a,anomaly_score)
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    # breakpoint()
    # xs = fill_blank(CHECK_TS, xs, total_ts)
    return xs 
def fill_blank(check_ts, labels, total_ts):
    def ts_generator():
        for t in total_ts:
            yield t

    def label_generator():
        for t, label in zip(check_ts, labels):
            yield t, label

    g_ts = ts_generator()
    g_label = label_generator()
    final_labels = []

    try:
        current = next(g_ts)
        ts_label, label = next(g_label)
        while True:
            if current > ts_label:
                ts_label, label = next(g_label)
                continue
            elif current < ts_label:
                final_labels.append(0)
                current = next(g_ts)
                continue
            final_labels.append(label)
            current = next(g_ts)
            ts_label, label = next(g_label)
    except StopIteration:
        return np.array(final_labels, dtype=np.int8)


def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=1)
    k = 10
    explainer = VGradExplainer(is_batched=True, r2b_method=TopKR2B(model.in_shape, k))
    model = nn.DataParallel(model)
    # explainer = PlainGradExplainer(0.25)
    y_true, y_pred, w_true, w_pred = [], [], [], []
    # with torch.no_grad():
    # cnt = 0
    for batch in tqdm(dataloader):
        # cnt +=1
        # if cnt == 3:
        #     break
        # breakpoint()
        x = batch["x"].cuda()
        # x = xx.view(xx.shape[1:])
        y = batch["y"].cuda()
        
        # alpha = torch.ones_like(xx)
        
        y_p = model(x)
        # breakpoint()
        # torch nn.LSTM doesn't allow backward in eval mode, so we make it train briefly
        # if isinstance(model, SimpleLSTM):
        model.train()
            # y, v, test = explainer._find_grad_order(model, x)
        
       
        w_p = explainer.get_explanation(model, x)
        
        # if isinstance(model, SimpleLSTM):
        model.eval()
        
        # ts.append(np.array(batch["ts"]))
        y_true.append(y.cpu())
        y_pred.append(y_p.max().cpu())
        w_true.append(batch["exp"][0].cpu())
        w_pred.append(w_p.cpu())
        gc.collect()
        torch.cuda.empty_cache()
        
        # breakpoint()

        # try:
        #     att.append(np.array(batch["attack"]))
        # except:
        #     att.append(np.zeros(batch_size))
        # breakpoint()     
    return (
        torch.stack(y_true),
        torch.stack(y_pred),
        torch.stack(w_true),
        torch.stack(w_pred)
    )    

if __name__ == '__main__':
    # model = LSTMModel()
    # model = train(model)
    ds_name = 'hai_sliding_100'
    ds_name_train = ds_name + "_train"
    ds_name_test = ds_name + "_test"
    model_name = "lstm"
    explainer = "VG"
    # train_dataset = get_dataset(ds_name_train)
    test_dataset = get_dataset(ds_name_test)
    d = test_dataset.num_features
    
    

    # # model = StackedLSTM()
 
    # model = SimpleLSTM(in_shape = (d,), out_shape = (1,)).double()
    # # breakpoint()
    # model = nn.DataParallel(model)
    
    # model.cuda()
    # model.train()
    # BEST_MODEL, LOSS_HISTORY = train(train_dataset, model, BATCH_SIZE, epoch) 
    

    # with open("models/LSTM/SimpleLSTM.pt", "wb") as f:
    #     torch.save(
    #         {
    #             "state": BEST_MODEL["state"],
    #             "best_epoch": BEST_MODEL["epoch"],
    #             "loss_history": LOSS_HISTORY,
    #         },
    #         f,
    #     )

    with open("models/LSTM/SimpleLSTM.pt", "rb") as f:
        SAVED_MODEL = torch.load(f)


    # MODEL = StackedLSTM()
    loaded_model = SimpleLSTM(in_shape = (d,), out_shape = (1,)).double()
    
    new = {k.replace('module.',''):v for k,v in SAVED_MODEL["state"].items()}
    loaded_model.load_state_dict(new)
    
    
    loaded_model.eval().cuda()
    y_true, y_pred, w_true, w_pred = inference(test_dataset, loaded_model, BATCH_SIZE) 
    
    
    
    acc, f1, fpr, fnr = summary(y_true.cpu(), y_pred.detach().cpu().numpy(), score = True)
    f = open(f"results/{ds_name}_experiment_result_model.txt", "a")
    f.write(f"{model_name} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\ \n")
    f.close()
    acc, f1, fpr, fnr = summary(w_true.cpu().flatten(), w_pred.detach().cpu().numpy().flatten(), score = False)
    f = open(f"results/{ds_name}_experiment_result_explainer.txt", "a")
    f.write(f"{model_name} | {explainer} & {fpr:.4f} & {fnr:.4f} & {acc:.4f} & {f1:.4f}     \\\\ \n")
    f.close()
    
    
    
    # breakpoint() 
    # ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)
    # np.savetxt('models/LSTM/StackedLSTM_test_anomaly_score.csv', ANOMALY_SCORE, delimiter=',') 
    # THRESHOLD = 0.008
    # y_pred = anomaly_detection(ANOMALY_SCORE, THRESHOLD, test_dataset.get_ts())
    # y_true = CHECK_ATT
    


    
    
    
    
    
    # acc, f1, fpr, fnr = summary(y_true, y_pred)
    
    
    
    # print(f"& {fpr:.2f} & {fnr:.2f} & {acc:.2f} & {f1:.2f}     \\")
    # print("-"*100)
    # print(f"fpr: {fpr:.2f}")
    # print(f"fnr: {fnr:.2f}")
    # print(f"acc: {acc:.2f}")
    # print(f"f1: {f1:.2f}")
    # print("-"*100)
    
    
    breakpoint()
    
    
        

    