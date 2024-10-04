import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import MinMaxScaler
import re
import dataProcess_BMI
import NetModel
import os
import time

# set random
# get train-set and targets
WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231212'  # setting work-folder
file_path = f'{WorkingFolder}/combinded.csv'  # define cutting space video
data = pd.read_csv(file_path)
train_cut = int(len(data)*0.7)
data_train = data.loc[:train_cut]
data_test = data.loc[train_cut:len(data)]

# generate folder
WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel'  # setting work-folder
NetDes = "R701D20231212_CNN(1DK1S1)_LSTM(L2H200)"
LocTime = time.strftime('%Y%m%d-%H%M', time.localtime())
Path = f'{WorkingFolder}/{LocTime}_{NetDes}'  # define cutting space video
os.makedirs(f'{Path}')

Time_step = 120
# trans pds to tensor
Input, Target = dataProcess_BMI.combined2tensor(data=data_train, inputTag="neu", targerTag="out", Time_step=Time_step, neuLim=False, agg_row=3)
Input_test, Target_test = dataProcess_BMI.combined2tensor(data=data_test, inputTag="neu", targerTag="out", Time_step=Time_step, neuLim=False, agg_row=3)
print(Input)
print(Target)

if torch.cuda.is_available():
    Input = Input.cuda()
    Target = Target.cuda()
    Input_test = Input_test.cuda()
    Target_test = Target_test.cuda()

# net parameters
num_epoch = 10000
batch_size = 900
conv_input = len(Input[0, :, 0])
hidden_size = 200
num_layers = 2
output_size = len(Target[0, :])
in_channels = batch_size
out_channel = batch_size
kernel_size = 1
stride = 1
conv_out = int(len(Input[0, 0, :])-kernel_size/stride) + 1
# Training model
model = NetModel.CNN_LSTM(conv_input, conv_out, kernel_size, stride, hidden_size, num_layers, output_size).cuda()
print(model)


# optimizer & loss
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
loss_func = nn.CrossEntropyLoss()
iters = int(len(Input)/batch_size)

# Training log
epoch_r = []
train_losses = []
Corr_rate_test = []
Corr_rate_train = []

for epoch in range(num_epoch):
    for interation in range(iters):
        start_pos = interation * batch_size
        end_pos = (interation + 1) * batch_size

        train_x1 = Input[start_pos:end_pos].cuda()
        train_y1 = Target[start_pos:end_pos].cuda()

        optimizer.zero_grad()
        output = model(train_x1)

        loss = loss_func(output, train_y1)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        epoch_r.append(epoch)
        model.eval()
        with torch.no_grad():
            output_test = model(Input_test)
            test_loss = loss_func(output_test, Target_test)
        output_test = output_test.cpu().data.numpy()
        Target_test_CPU = Target_test.cpu()
        correct_count = 0
        for m in range(len(Target_test_CPU)):
            max1 = np.argmax(output_test[m, :])
            max2 = np.argmax(Target_test_CPU[m, :])
            if max1 == max2:
                correct_count += 1
        rate_test = correct_count / len(Target_test_CPU)
        Corr_rate_test.append(rate_test)

        with torch.no_grad():
            output_train = model(Input)
            test_loss = loss_func(output_train, Target)
            test_loss.cpu().data.numpy()
            train_losses.append(test_loss)
        output_train = output_train.cpu().data.numpy()
        Target_CPU = Target.cpu()
        correct_count = 0
        for m in range(len(Target_CPU)):
            max1 = np.argmax(output_train[m, :])
            max2 = np.argmax(Target_CPU[m, :])
            if max1 == max2:
                correct_count += 1
        rate_train = correct_count / len(Target_CPU)
        Corr_rate_train.append(rate_train)
        print("epoch{},test_loss:{}, train_rate{},test_rate{}".format(epoch, test_loss, rate_train, rate_test))
        model.train()

# generating training log
train_log = [epoch_r, train_losses, Corr_rate_test, Corr_rate_train]
train_log = pd.DataFrame(train_log).transpose()
train_log.columns = ['epoch', "loss", "Corr_rate_test", "Corr_rate_train"]
train_log.to_csv(f'{Path}/log.csv')
torch.save(model, f"{Path}/seqnet.pkl")

