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

    def forward(self, x, alpha):
        # x = x[:,:,LEAV_IDX] # batch, window_size, params
        N, W, d = x.shape
        # print(alpha.shape)
        x = x * alpha
        
        avg_x = x.mean(dim=1).view(N, d, 1)
        attention = avg_x
        connection = attention
        connection = connection.reshape(-1, d) # batch, params
        
       
        attention = self.relu(torch.squeeze(attention))
        attention = self.relu(self.dense1(attention))
        attention = self.sigmoid(self.dense2(attention)) 

        x = x.transpose(0, 1)  # (batch, window_size, params) -> (window_size, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(self.relu(outs[-1])) 

        mix_factor = self.sigmoid(self.w) 
        # return connection
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