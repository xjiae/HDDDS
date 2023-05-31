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

from tqdm.notebook import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
from models.lstm import StackedLSTM
from utils import *
from explainers.intgrad import PlainGradExplainer, IntGradExplainer

N_HIDDENS = 200
N_LAYERS = 3
BATCH_SIZE = 2024
epoch = 130
num_features = 86

def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()
    # epochs = trange(n_epochs, desc="training")
    best = {"loss": sys.float_info.max}
    loss_history = []

    for e in tqdm(range(n_epochs)):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x = batch["x"].cuda()
            xl= batch["xl"].cuda()
            
            alpha = torch.ones_like(x)
            
            
            y = model(x, alpha)
            loss = loss_fn(xl, y) 

            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            break
        
        loss_history.append(epoch_loss)
        # epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1
        break
        
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
    explainer = PlainGradExplainer(0.25)
    ts, dist, att = [], [], []
    # with torch.no_grad():

    for batch in dataloader:
    
        xx = batch["x"].cuda()
        x = xx.view(xx.shape[1:])
        y = batch["y"].cuda()
        
        alpha = torch.ones_like(xx)
        guess = model(xx, alpha)
        if isinstance(model, StackedLSTM):
            model.train()
            y, v, test = explainer._find_grad_order(model, x)
        
       
        alpha_pred = explainer.find_explanation(model, x)
        # torch nn doesn't allow backward in eval mode
        if isinstance(model, StackedLSTM):
            model.eval()
        
        ts.append(np.array(batch["ts"]))
        dist.append(torch.abs(y - guess).cpu().numpy())

        try:
            att.append(np.array(batch["attack"]))
        except:
            att.append(np.zeros(batch_size))
            
    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(att),
    )    

if __name__ == '__main__':
    # model = LSTMModel()
    # model = train(model)
   
    train_dataset = get_dataset('hai_sliding_train_100')
    
    

    model = StackedLSTM()
    # breakpoint()
    model = nn.DataParallel(model)
    
    model.cuda()
    model.train()
    # BEST_MODEL, LOSS_HISTORY = train(train_dataset, model, BATCH_SIZE, epoch) 
    

    # with open("models/LSTM/StackedLSTM_alpha.pt", "wb") as f:
    #     torch.save(
    #         {
    #             "state": BEST_MODEL["state"],
    #             "best_epoch": BEST_MODEL["epoch"],
    #             "loss_history": LOSS_HISTORY,
    #         },
    #         f,
    #     )

    with open("models/LSTM/StackedLSTM_alpha.pt", "rb") as f:
        SAVED_MODEL = torch.load(f)


    MODEL = StackedLSTM()
    new = {k.replace('module.',''):v for k,v in SAVED_MODEL["state"].items()}
    MODEL.load_state_dict(new)
    test_dataset = get_dataset('hai_sliding_test_100')
    
    MODEL.eval().cuda()
    CHECK_TS, CHECK_DIST, CHECK_ATT = inference(test_dataset, MODEL, BATCH_SIZE)  
    ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)
    np.savetxt('models/LSTM/StackedLSTM_test_anomaly_score.csv', ANOMALY_SCORE, delimiter=',') 
    THRESHOLD = 0.008
    y_pred = anomaly_detection(ANOMALY_SCORE, THRESHOLD, test_dataset.get_ts())
    y_true = CHECK_ATT

    
    
    
    
    
    # acc, f1, fpr, fnr = summary(y_true, y_pred)
    
    
    
    # print(f"& {fpr:.2f} & {fnr:.2f} & {acc:.2f} & {f1:.2f}     \\")
    # print("-"*100)
    # print(f"fpr: {fpr:.2f}")
    # print(f"fnr: {fnr:.2f}")
    # print(f"acc: {acc:.2f}")
    # print(f"f1: {f1:.2f}")
    # print("-"*100)
    
    
    breakpoint()
    
    
        

    