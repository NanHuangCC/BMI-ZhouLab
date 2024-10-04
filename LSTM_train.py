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
# Rat = "R905"
# Date = "20240122"
# get train-set and targets
# WorkingFolder = fr'Z:\Project-NC-2023-A-02\1 - data\Rats\{Rat}\{Date}'  # setting work-folder
# file_path = f'{WorkingFolder}/combinded.csv'  # define cutting space video
WorkingFolder = fr'Z:\Project-NC-2023-A-02\1 - data\People\nan\KeyData'  # setting work-folder
file_path = f'{WorkingFolder}/NeuAve_train_set.csv'
data = pd.read_csv(file_path)
train_cut = int(len(data)*0.9)
train_cut1 = int(len(data)*1)
data_train = data.loc[:train_cut]
data_test = data.loc[train_cut:train_cut1]

# generate folder
WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel'  # setting work-folder
NetDes = f"Overall1_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F"
LocTime = time.strftime('%Y%m%d-%H%M', time.localtime())
Path = f'{WorkingFolder}/{LocTime}_{NetDes}'  # define cutting space video
os.makedirs(f'{Path}')
print("data loaded")

Time_step = 300
# trans pds to tensor
Input, Target = dataProcess_BMI.combined2tensor(data=data_train, inputTag="NeuAve", targerTag="out", Time_step=Time_step, neuLim=False, agg_row=6)
Input_test, Target_test = dataProcess_BMI.combined2tensor(data=data_test, inputTag="NeuAve", targerTag="out", Time_step=Time_step, neuLim=False, agg_row=6)
print(Input)
print(Target)
print("Transfer finished")

if torch.cuda.is_available():
    Input = Input.cuda()
    Target = Target.cuda()
    Input_test = Input_test.cuda()
    Target_test = Target_test.cuda()

# net parameters
num_epoch = 2000
batch_size = 900
conv_input = len(Input[0, :, 0])
hidden_size = 200
num_layers = 2
output_size = len(Target[0, :])
in_channels = batch_size
out_channel = batch_size
kernel_size = 1
stride = 1
conv_out = int(len(Input[0, :, 0])-kernel_size/stride) + 1
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
Test_losses = []
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

    if epoch % 20 == 0:
        epoch_r.append(epoch)
        model.eval()
        with torch.no_grad():
            output_test = model(Input_test)
            test_loss = loss_func(output_test, Target_test)
            test_loss.cpu().data.numpy()
            Test_losses.append(test_loss)
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
            Train_loss = loss_func(output_train, Target)
            Train_loss.cpu().data.numpy()
            train_losses.append(Train_loss)
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
        print("epoch{},train_loss:{},test_loss:{}, train_rate{},test_rate{}".format(epoch, Train_loss,test_loss, rate_train, rate_test))
        model.train()

# generating training log
train_log = [epoch_r, train_losses, Test_losses, Corr_rate_train, Corr_rate_test]
train_log = pd.DataFrame(train_log).transpose()
train_log.columns = ['epoch', "train_loss", "test_loss", "Corr_rate_train", "Corr_rate_test"]
train_log.to_csv(f'{Path}/log.csv')
torch.save(model, f"{Path}/seqnet.pkl")
