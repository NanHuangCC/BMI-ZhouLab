import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import NetModel
import pandas as pd
import time
import os
import re
import warnings
import dataProcess_BMI
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
NetFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240111-1425_clusterSize4000_Neuron1000_CNN'  # setting work-folder
netPath = f'{NetFolder}/20240111_CNN.pkl'  # define net path

WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240122\Behavioral'  # setting work-folder
Input_path = f'{WorkingFolder}/Full_Inf.csv'  # define net path

# reload net
net = torch.load(netPath).cuda()
print(net)
# load input data
Input = dataProcess_BMI.pd2tensor(Input_path)
scaler = MinMaxScaler()
Zscore = StandardScaler()
Input = Zscore.fit_transform(Input)
Input = scaler.fit_transform(Input)
Input = torch.from_numpy(Input).type(torch.FloatTensor)
print(Input)


# net calculation
y = net(Input)

# output result
BehavioralIndex = y.cpu().data.numpy()

maxList = []
for m in range(BehavioralIndex.shape[0]):
    if np.max(BehavioralIndex[m, :]) > 0.75:
        max1 = np.argmax(BehavioralIndex[m, :])
        maxList.append(max1)
    else:
        maxList.append(-1)


maxList = np.array(maxList)
BehavioralIndex = np.c_[BehavioralIndex, maxList]
BehavioralIndex = pd.DataFrame(BehavioralIndex)

colnames = []
for m in range((BehavioralIndex.shape[1]-1)):
    name = f"out{m}"
    colnames.append(name)
colnames.append("labels")
BehavioralIndex.columns = colnames

BehavioralIndex.insert(loc=len(BehavioralIndex.columns), column='Description', value="None")

BehavioralIndex.Description[BehavioralIndex.labels == -1] = "Uncertain"
BehavioralIndex.Description[BehavioralIndex.labels == 0] = "Resting"
BehavioralIndex.Description[BehavioralIndex.labels == 1] = "Drinking"
BehavioralIndex.Description[BehavioralIndex.labels == 2] = "Stepping"
BehavioralIndex.Description[BehavioralIndex.labels == 3] = "turning"
BehavioralIndex.Description[BehavioralIndex.labels == 4] = "standing"
BehavioralIndex.Description[BehavioralIndex.labels == 5] = "grooming"


BehavioralIndex.to_csv(f'{WorkingFolder}/BehavioralIndex.csv')