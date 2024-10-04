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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# Set analysis session ----------------------- Read data
Rat = "R701"
Date = "20231129"

WorkingFolder = rf'Z:\Project-NC-2023-A-02\1 - data\0data\MPrediction\{Rat}_{Date}'   # setting work-folder
Beh_path = f'{WorkingFolder}/RawData/Behavior.csv'  # behavioral path
Neu_path = f'{WorkingFolder}/RawData/Neu.csv'  # Neural path

# generate dir for saving result
SavePath = f'{WorkingFolder}/Result'
if os.path.exists(SavePath):
    print(f'exists:{SavePath}')
else:
    os.makedirs(f'{SavePath}')

# Input csv path get tensor ###########################################################################################
Ust, Umt, Xt, Yt = dataProcess_BMI.ReadBehNeu(Beh_path= Beh_path, Neu_path= Beh_path, TauT= 3, FramePerS= 5)

# generate shuffle IDX
RamIdx1 = torch.randperm(Xt.shape[0])
RamIdx2 = torch.randperm(Xt.shape[0])
RamIdx3 = torch.randperm(Xt.shape[0])
# Chose Neuron to predict
NeuID = 10
NeuPath = f'{SavePath}/Neuron{NeuID}'
if os.path.exists(NeuPath):
    print(f'exists:{NeuPath}')
else:
    os.makedirs(f'{NeuPath}')

NetPath = f'{NeuPath}/NetModel'
if os.path.exists(NetPath):
    print(f'exists:{NetPath}')
else:
    os.makedirs(f'{NetPath}')









Xt = Xt[:, NeuID:(NeuID+1)]
Yt = Yt[:, NeuID:(NeuID+1)]
# select shuffle part #################################################################################################
ShuffleType = 0

if ShuffleType == 1:
    Xt = Xt[RamIdx1, :]

if ShuffleType == 2:
    Ust = Ust[RamIdx1, :, :, :]

if ShuffleType == 3:
    Umt = Umt[RamIdx1, :, :, :]

if ShuffleType == 4:
    Ust = Ust[RamIdx1, :, :, :]
    Umt = Umt[RamIdx2, :, :, :]

if ShuffleType == 5:
    Ust = Ust[RamIdx1, :, :, :]
    Umt = Umt[RamIdx2, :, :, :]
    Xt = Xt[RamIdx3, :]



# Setting train set & Test set ########################################################################################
# 12min time a time
# Set  train range
timerange = 0
StartTrain = timerange * 3600
EndTrain = StartTrain + 3000
EndTest = EndTrain + 600

Xt_train = Xt[StartTrain:EndTrain, :]
Xt_test = Xt[EndTrain: EndTest, :]
Yt_train = Yt[StartTrain:EndTrain, :]
Yt_test = Yt[EndTrain: EndTest, :]
Ust_train = Ust[StartTrain:EndTrain, :, :, :]
Ust_test = Ust[EndTrain: EndTest, :, :, :]
Umt_train = Umt[StartTrain:EndTrain, :, :, :]
Umt_test = Umt[EndTrain: EndTest, :, :, :]

# Net work Parameters #################################################################################################
# 2D-CNN
in_channels = 1
out_channel = 1
kernel_size = (int(3), int(3))
stride = (3, 3)
# 1D-CNN
TauFrame = 15
in_channels1D = int(TauFrame / 3)
out_channels1D = int(1)
kernel_size1D = int(1)
stride1D = int(1)
# Linear
input_size = int(Xt.shape[1] + (Ust.shape[3] / 3) + (Umt.shape[3] / 3))
hidden_size = int(512)
outsize = int(Xt.shape[1])
# Network define ######################################################################################################
model = NetModel.NeuActRegNET(in_channels2DS=1, out_channel2DS=1, kernel_size2DS=kernel_size, stride2DS=stride,
                                  in_channels1DS=in_channels1D, out_channels1DS=out_channels1D,
                                  kernel_size1DS=kernel_size1D, stride1DS=stride1D,
                                  in_channels2DM=1, out_channel2DM=1, kernel_size2DM=kernel_size, stride2DM=stride,
                                  in_channels1DM=in_channels1D, out_channels1DM=out_channels1D,
                                  kernel_size1DM=kernel_size1D, stride1DM=stride1D,
                                  DeepFeatures=input_size, hidden_size=hidden_size, outsize=outsize)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # use a optimizer
loss_func = nn.MSELoss()  # setting a loss function
print(model)

# pull all data into cuda
if torch.cuda.is_available():
    Xt_train = Xt_train.cuda()
    Xt_test = Xt_test.cuda()
    Yt_train = Yt_train.cuda()
    Yt_test = Yt_test.cuda()
    Ust_train = Ust_train.cuda()
    Ust_test = Ust_test.cuda()
    Umt_train = Umt_train.cuda()
    Umt_test = Umt_test.cuda()
    model = model.cuda()

# Training log setting
num_epoch = 2000
epoch_r = []
losses_train = []
losses_test = []
MSEs_train = []
MSEs_test = []
# Train Network #######################################################################################################
for epoch in range(num_epoch):
    model.train()
    out, a = model(Ust_train, Umt_train, Xt_train)  # input x
    loss = loss_func(out, Yt_train)  # calculate loss
    optimizer.zero_grad()
    loss.backward()  # BP process
    optimizer.step()  # optimize
    if epoch % 200 == 0:
        epoch_r.append(epoch)
        model.eval()
        with torch.no_grad():
            # Train Set
            output_train, b = model(Ust_train, Umt_train, Xt_train)
            loss_train = loss_func(output_train, Yt_train)
            # Test Set
            output_test, c = model(Ust_test, Umt_test, Xt_test)
            loss_test = loss_func(output_test, Yt_test)
            loss_test.cpu().data.numpy()
            loss_train.cpu().data.numpy()
            losses_test.append(loss_test)
            losses_train.append(loss_train)

            # MSE train set
        output_train = output_train.cpu().data.numpy()
        Yt_train_CPU = Yt_train.cpu()
        MSE_train = mean_squared_error(output_train, Yt_train_CPU)
        MSEs_train.append(MSE_train)
        # MSE Test set
        output_test = output_test.cpu().data.numpy()
        Yt_test_CPU = Yt_test.cpu()
        MSE_test = mean_squared_error(output_test, Yt_test_CPU)
        MSEs_test.append(MSE_test)

        print("epoch{},train_loss:{},test_loss:{}, MSE_train{},MSE_test{}".format(epoch, loss_train, loss_test,
                                                                                      MSE_train, MSE_test))


# Saving training log ##################################################################################################
train_log = [epoch_r, losses_train, losses_test, MSEs_train, MSEs_test]
train_log = pd.DataFrame(train_log).transpose()
train_log.columns = ['epoch', "losses_train", "losses_test", "MSEs_train", "MSEs_test"]
train_log.to_csv(f'{NetPath}/log.csv')
torch.save(model, f"{NetPath}/model.pkl")

# Get full result
model.eval()
model = model.cpu()
with torch.no_grad():
    output, DeepFeature = model(Ust, Umt, Xt)

# Background range : all train set
# T1 = np.arange(0, 3000)
# T2 = np.arange(3600, 6600)
# T3 = np.arange(7200, 10200)
# T4 = np.arange(10800, 13800)
# BGrange = np.concatenate((T1,T2,T3,T4), axis=0)

# T1 = np.arange(3000, 3600)
# T2 = np.arange(6600, 7200)
# T3 = np.arange(10200, 10800)
# T4 = np.arange(13800, 14400)
# Trange = np.concatenate((T1,T2,T3,T4), axis=0)

T1 = np.arange(0, 30)
T2 = np.arange(36, 66)
T3 = np.arange(72, 102)
T4 = np.arange(108, 138)
BGrange = np.concatenate((T1,T2,T3,T4), axis=0)

T1 = np.arange(30, 36)
T2 = np.arange(66, 72)
T3 = np.arange(102, 108)
T4 = np.arange(138, 144)
Trange = np.concatenate((T1,T2,T3,T4), axis=0)


print(f"Shap explain")
ExpNet = model.LinNet

background = DeepFeature[BGrange, ]
TestSet = DeepFeature[Trange, ]
explainer = shap.DeepExplainer(model=ExpNet, data=background)
shap_values = explainer.shap_values(TestSet, ranked_outputs=None)

print(type(shap_values))
print(shap_values.shape)
for i in range(shap_values.shape[2]):
    np.savetxt(f'{NeuPath}/Shap{i}.csv', shap_values[:, :, i], delimiter=",")


output.data.numpy()
DeepFeature.data.numpy()
Full_result = np.c_[DeepFeature, output]
Full_result = pd.DataFrame(Full_result)
Full_result.to_csv(f'{NeuPath}/model_result.csv')

