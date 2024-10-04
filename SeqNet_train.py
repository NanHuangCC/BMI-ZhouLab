import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import NetModel
import pandas as pd
import time
import os
import re
import dataProcess_BMI

if __name__ == "__main__":
    # read train data
    WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\BehavioralSet\BehavioralFeature'  # setting work-folder
    Path = f'{WorkingFolder}/Full_umap_newlabels.csv'  # define cutting space video
    x, y = dataProcess_BMI.get_trainSet(Path, clusterSize=4000, labelname="labels2")

    # generate folder
    WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel'  # setting work-folder
    NetDes = "clusterSize4000_Neuron1000_DropOut_CNN"
    LocTime = time.strftime('%Y%m%d-%H%M', time.localtime())
    Path = f'{WorkingFolder}/{LocTime}_{NetDes}'  # define cutting space video
    os.makedirs(f'{Path}')

    # para for iter
    samplenum = x.shape[0]
    input_size = x.shape[1]
    output_size = y.shape[1]
    minibatch = x.shape[0]

    # use rnn model
    net = NetModel.SeqNET(input_size=input_size,
                          outsize=output_size,
                          hidden_size=1000).cuda()
    # setting net parameters
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # optim
    loss_func = nn.CrossEntropyLoss()
    print(net)

    epoch_r = []
    step_r = []
    correct_r = []
    loss_r = []
    for epoch in range(10000):
        for iterations in range(int(samplenum / minibatch)):
            start = iterations * minibatch
            end = (iterations+1) * minibatch
            x0 = x[start:end, :].cuda()
            y0 = y[start:end, :].cuda()
            out = net(x0)
            loss = loss_func(out, y0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                loss_tmp = loss.cpu().data.numpy()
                out_tmp = out.cpu().data.numpy().reshape(minibatch, output_size)
                target_tmp = y0.cpu().data.numpy().reshape(minibatch, output_size)
                correct_count = 0
                for m in range(minibatch):
                    max1 = np.argmax(out_tmp[m, :])
                    max2 = np.argmax(target_tmp[m, :])
                    if max1 == max2:
                        correct_count += 1
                rate = correct_count/minibatch
                print(f"epoch:{epoch}, iterations:{iterations}, correctRate:{rate*100}%,loss:{loss_tmp}")
                epoch_r.append(epoch)
                step_r.append(iterations)
                correct_r.append(rate)
                loss_r.append(loss_tmp)
    # generating training log
    train_log = [epoch_r, step_r, correct_r, loss_r]
    train_log = pd.DataFrame(train_log).transpose()
    train_log.columns = ['epoch', "step", "correct", "loss"]
    train_log.to_csv(f'{Path}/log.csv')
    torch.save(net, f"{Path}/seqnet.pkl")


