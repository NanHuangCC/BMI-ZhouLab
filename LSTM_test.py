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


NetPath = [
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1232_R70120231129_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1233_R70120231130_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1235_R70120231207_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1236_R70120231212_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1237_R70120231215_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1239_R70120231219_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1247_R70120231221_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1248_R70120231222_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1308_R90220240104_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1309_R90220240106_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1310_R90220240111_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1313_R90220240116_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1314_R90220240117_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1317_R90220240119_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1318_R90220240120_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1320_R90220240121_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1321_R90520240117_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1322_R90520240118_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1323_R90520240119_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1324_R90520240120_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1325_R90520240121_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F',
    r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240322-1326_R90520240122_NeuAve10_CNN(1DK1S1)_LSTM(L2H200)_F'
]

BehPath = [
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231129',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231130',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231207',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231212',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231215',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231219',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231221',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R701\20231222',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240104',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240106',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240111',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240116',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240117',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240119',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240120',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R902\20240121',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R905\20240117',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R905\20240118',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R905\20240119',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R905\20240120',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R905\20240121',
    r'Z:\Project-NC-2023-A-02\1 - data\Rats\R905\20240122'
]

logfolder = r"Z:\Project-NC-2023-A-02\1 - data\0data\summary\20240322\outputCrossAnimal"

mat = np.zeros((len(NetPath),len(BehPath)), dtype='float32')

for netmodel in range(len(NetPath)):
    netPath = f'{NetPath[netmodel]}/seqnet.pkl'  # define net path
    for data_Set in range(len(BehPath)):
        file_path = f'{BehPath[data_Set]}/combinded.csv'  # define net path
        data = pd.read_csv(file_path)
        train_cut = int(len(data) * 0.6)
        train_cut1 = int(len(data) * 1)
        data_test = data.loc[train_cut:train_cut1]

        Time_step = 300
        Input_test, Target_test = dataProcess_BMI.combined2tensor(data=data_test, inputTag="NeuAve", targerTag="out",
                                                                  Time_step=Time_step, neuLim=False, agg_row=6)
        if torch.cuda.is_available():
            Input_test = Input_test.cuda()
            Target_test = Target_test.cuda()
            softmax = nn.Softmax(dim=1).cuda()
            model = torch.load(netPath).cuda()

        model.eval()
        output_test = model(Input_test)
        output_test = softmax(output_test)
        output_test = output_test.cpu().data.numpy()
        Target_test_CPU = Target_test.cpu()
        correct_count = 0
        for m in range(len(Target_test_CPU)):
            max1 = np.argmax(output_test[m, :])
            max2 = np.argmax(Target_test_CPU[m, :])
            if max1 == max2:
                correct_count += 1
        rate_test = correct_count / len(Target_test_CPU)
        mat[netmodel, data_Set] = rate_test
        output_test_CSV = pd.DataFrame(output_test)
        Target_test_CSV = pd.DataFrame(Target_test_CPU)
        print(mat)

        output_test_CSV.to_csv(f'{logfolder}/out{netmodel}_{data_Set}.csv')
        Target_test_CSV.to_csv(f'{logfolder}/train{netmodel}_{data_Set}.csv')

mat_CSV = pd.DataFrame(mat)
mat_CSV.to_csv(f'{logfolder}/summary_mat.csv')
