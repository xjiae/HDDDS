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

from utils import *

N_HIDDENS = 200
N_LAYERS = 3
BATCH_SIZE = 2024
epoch = 130
num_features = 86
class StackedLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0.1
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2,num_features)
        self.relu = torch.nn.LeakyReLU(0.1)
        
  
        w = torch.nn.Parameter(torch.FloatTensor([-0.01]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        
        self.sigmoid = torch.nn.Sigmoid()
        

        self.dense1 = torch.nn.Linear(num_features, num_features//2)
        self.dense2 = torch.nn.Linear(num_features//2, num_features)

    def forward(self, x):
        # x = x[:,:,LEAV_IDX] # batch, window_size, params

        pool = torch.nn.AdaptiveAvgPool1d(1)
        
        attention_x = x
        attention_x = attention_x.transpose(1,2) # batch, params, window_size
        
        attention = pool(attention_x) # batch, params, 1
        
        connection = attention
        connection = connection.reshape(-1,num_features) # batch, params
        
       
        attention = self.relu(torch.squeeze(attention))
        attention = self.relu(self.dense1(attention))
        attention = self.sigmoid(self.dense2(attention)) 

        x = x.transpose(0, 1)  # (batch, window_size, params) -> (window_size, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(self.relu(outs[-1])) 

        mix_factor = self.sigmoid(self.w) 

        return mix_factor * connection * attention + out * (1 - mix_factor) 

class LSTMModel(nn.Module):
    def __init__(self, n_features=86):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=100, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=200, hidden_size=100, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=200, hidden_size=100, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(200, n_features)

    def forward(self, inputs):
        first, _ = self.lstm1(inputs)
        second, _ = self.lstm2(first)
        third, _ = self.lstm3(second)
        lstm_out = self.dense(third)
        # outputs = lstm_out + aux_input
        return lstm_out

def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()
    # epochs = trange(n_epochs, desc="training")
    best = {"loss": sys.float_info.max}
    loss_history = []
    for e in tqdm(range(n_epochs)):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            guess = model(given)
            loss = loss_fn(answer[:,], guess) 
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        loss_history.append(epoch_loss)
        # epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1
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
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist, att = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            guess = model(given)
            ts.append(np.array(batch["ts"]))
            dist.append(torch.abs(answer - guess).cpu().numpy())

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
   
    # train_dataset = get_dataset('hai')

    # model = StackedLSTM()
    # model = nn.DataParallel(model)
    
    # model.cuda()
    # model.train()
    # BEST_MODEL, LOSS_HISTORY = train(train_dataset, model, BATCH_SIZE, epoch) 
    

    # with open("models/LSTM/StackedLSTM.pt", "wb") as f:
    #     torch.save(
    #         {
    #             "state": BEST_MODEL["state"],
    #             "best_epoch": BEST_MODEL["epoch"],
    #             "loss_history": LOSS_HISTORY,
    #         },
    #         f,
    #     )
    # breakpoint()
    with open("models/LSTM/StackedLSTM.pt", "rb") as f:
        SAVED_MODEL = torch.load(f)


    MODEL = StackedLSTM()
    new = {k.replace('module.',''):v for k,v in SAVED_MODEL["state"].items()}
    MODEL.load_state_dict(new)
    test_dataset = get_dataset('hai_test')
    MODEL.eval().cuda()
    CHECK_TS, CHECK_DIST, CHECK_ATT = inference(test_dataset, MODEL, BATCH_SIZE)  
    ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)
    np.savetxt('models/LSTM/StackedLSTM_test_anomaly_score.csv', ANOMALY_SCORE, delimiter=',') 
    THRESHOLD = 0.008
    y_pred = anomaly_detection(ANOMALY_SCORE, THRESHOLD, test_dataset.get_ts())
    y_true = CHECK_ATT
    
    
    acc, f1, fpr, fnr = summary(y_true, y_pred)
    print("-"*100)
    print(f"fpr: {fpr:.2f}")
    print(f"fnr: {fnr:.2f}")
    print(f"acc: {acc:.2f}")
    print(f"f1: {f1:.2f}")
    print("-"*100)
    
    
    breakpoint()
    
    
        

    # # Extract features and labels from the testing dataloader
    # test_features = []
    # test_labels = []
    # test_explanations = []

    # # Iterate over the test dataloader
    # for batch in test_dataloader:
    #     if cnt == 4:
    #         break
    #     # Access the batch data
    #     features = normalizer.transform(batch[0])  # X
    #     labels = batch[1]   # y
    #     explanations = batch[2]  # explanation
    #     test_features.append(features)
    #     test_labels.append(labels)
    #     test_explanations.append(explanations)
    #     cnt += 1


    # train_features = np.vstack(train_features)
    # test_features = np.vstack(test_features)
    # test_labels = np.hstack(test_labels)

    # # Predict the labels for the testing data
    # test_predictions = svm.predict(test_features)

    # test_predictions[np.where(test_predictions==-1)] = 0
    # # Evaluate the performance of the One-Class SVM
    # accuracy = accuracy_score(test_labels, test_predictions)

    # # Print the accuracy score
    # print('Accuracy:', accuracy)
    # explainer = shap.KernelExplainer(svm.decision_function, train_features[:10])
    # shap_values = explainer.shap_values(test_features)
    # test_explanations = np.vstack(test_explanations)

    # test_explanations = normalizer.fit_transform(test_explanations)



    # # Assuming you have 'shap_values' and 'test_predictions' arrays

    # thresholds = np.linspace(0, 1, num=100)  # Threshold values between 0 and 1
    # best_threshold = None
    # best_f1_score = 0.0
    # mean_fpr = 0.0
    # mean_fnr = 0.0

    # for i in tqdm(range(len(shap_values))):

    #     row = shap_values[i]
    #     f1_scores = []
    #     fprs = []
    #     fnrs = []

    #     for threshold in thresholds:
    #         binary_row = threshold_binary(row, threshold)
    #         f1 = f1_score(test_explanations[i], binary_row)
    #         f1_scores.append(f1)

    #     best_threshold = thresholds[np.argmax(f1_scores)]
    #     best_binary_row = threshold_binary(row, best_threshold)
    #     fpr, fnr = compute_fpr_fnr(test_explanations[i], best_binary_row)
    #     mean_fpr += fpr
    #     mean_fnr += fnr

    # mean_fpr /= len(shap_values)
    # mean_fnr /= len(shap_values)

    # print('Best threshold:', best_threshold)
    # print('Mean FPR:', mean_fpr)
    # print('Mean FNR:', mean_fnr)
    # breakpoint()
