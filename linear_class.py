#It's recommended to use this code in a Notebook environment for visualisation reasons

import os
import pandas as pd
import numpy as np
import string

import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import shuffle

np.random.seed(s)
torch.manual_seed(s)

nEpochs = 400
XY = data[data['LANG'].isin(['en','ge','it','du','sp','ru','ko'])]
XY.loc[:,'LANG'] = XY['LANG'].astype('category')
XY.loc[:,'LANG_'] = XY['LANG'].cat.codes

class linearClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(linearClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, input):
        return self.linear(input)

model = linearClassifier(4,7)
lossF = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())

class gaze_data(Dataset):
    def __init__(self,df):

        x=df[['NFIX','DUR','FF_DUR','MEAN_DUR']].values
        #x=df[['NRUN','REREAD','NFIX','REFIX','DUR','FF_SKIP','FF_NFIX','FF_DUR','MEAN_DUR','FF_MEAN_DUR']].values
        y=df['LANG_'].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

XY = shuffle(XY)
split = int(XY.shape[0]*0.9)
train_dataset = gaze_data(XY.iloc[:split,:])
test_dataset = gaze_data(XY.iloc[split:,:])

train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=36, shuffle=True)

f1_scoresp = []
precision_scoresp = []
recall_scoresp = []

for idx, (inp, outp) in enumerate(test_loader):
    out = model(inp)
    f1_scoresp.append(f1_score(outp.type(torch.LongTensor), torch.argmax(out,dim=1), average=None, labels=[0,1,2,3,4,5,6]))
    precision_scoresp.append(precision_score(outp.type(torch.LongTensor), torch.argmax(out,dim=1), average=None, labels=[0,1,2,3,4,5,6]))
    recall_scoresp.append(recall_score(outp.type(torch.LongTensor), torch.argmax(out,dim=1), average=None, labels=[0,1,2,3,4,5,6]))

loss_list = []
for _ in range(nEpochs):
    for idx, (inp, outp) in enumerate(train_loader):
        out = model(inp)
        loss = lossF(out, outp.type(torch.LongTensor))
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_list.append(loss.item())

f1_scores = []
precision_scores = []
recall_scores = []
for idx, (inp, outp) in enumerate(test_loader):
    out = model(inp)
    f1_scores.append(f1_score(outp.type(torch.LongTensor), torch.argmax(out,dim=1), average=None, labels=[0,1,2,3,4,5,6]))
    precision_scores.append(precision_score(outp.type(torch.LongTensor), torch.argmax(out,dim=1), average=None, labels=[0,1,2,3,4,5,6]))
    recall_scores.append(recall_score(outp.type(torch.LongTensor), torch.argmax(out,dim=1), average=None, labels=[0,1,2,3,4,5,6]))
