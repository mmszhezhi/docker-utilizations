import glob
import torch
import torch.nn as nn
import os,math
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

offset = 15000
path = "D:\d\pytorch-repo\docker-utilizations\model1"


df  = pd.read_csv("data.csv",index_col=0)
scaler = MinMaxScaler(feature_range=(-1, 1))


def create_inout_sequences(input_data, tw):
    seqs = []
    targets = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        seqs.append(train_seq)
        targets.append( train_label)
    return seqs,targets

class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=56, step=150):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(2, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, 2)
        self.step=step

    def forward(self, input_seq):
        h1 = torch.zeros(1, self.step, self.hidden_layer_size)
        c1 = torch.zeros(1, self.step, self.hidden_layer_size)

        lstm_out, self.hidden_cell = self.lstm(input_seq,(h1,c1))
        predictions = self.linear(lstm_out)
        return predictions[:,-1,:]


# bsize,step,features = 99,150,2

def genbatch(seqs,targets,bsize = 50):
    l = len(targets)
    for i in range(math.floor(l/bsize)):
        xb = seqs[i * bsize:(i + 1) * bsize]
        yb = targets[i * bsize:(i + 1) * bsize]
        yield torch.FloatTensor(np.stack(xb,axis=0)),torch.FloatTensor(np.stack(yb))

