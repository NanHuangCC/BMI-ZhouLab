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
    WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\test\BeahaviorCluster'  # setting work-folder
    Path = f'{WorkingFolder}/Full_umap.csv'  # define cutting space video
    trainSet = 20*60*30
    x, y = dataProcess_BMI.get_labels(Path, 1, trainSet)

    # parameter define
    Time_step = 20*60*30
    LR = 0.02

    # generate folder
    WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel'  # setting work-folder
    NetDes = "test"
    LocTime = time.strftime('%Y%m%d-%H%M', time.localtime())
    Path = f'{WorkingFolder}/{LocTime}_{NetDes}'  # define cutting space video
    os.makedirs(f'{Path}')

    # use rnn model
    net = NetModel.Rnn(input_size=x.shape[2],
                       outsize=y.shape[2],
                       num_layers=3, hidden_size=1000).cuda()
    print(net)

    # setting net parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # optim
    loss_func = nn.MSELoss()

    h_state = None

    epoch_r = []
    step_r = []
    correct_r = []
    loss_r = []
    for epoch in range(500):
        for step in range(int(trainSet / Time_step)):
            startPoint = (step * Time_step)
            EndPoint = ((step * Time_step) + Time_step)
            x_tmp = x[:, startPoint:EndPoint, :].cuda()
            y_tmp = y[:, startPoint:EndPoint, :].cuda()

            out, h_state = net(x_tmp, h_state)
            h_state = h_state.data
            loss = loss_func(out, y_tmp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate net para
            if epoch % 10 == 0:
                loss_tmp = loss.cpu().data.numpy()
                out_tmp = out.cpu().data.numpy().reshape(Time_step,y.shape[2])
                target_tmp = y_tmp.cpu().data.numpy().reshape(Time_step,y.shape[2])
                correct_count = 0
                for m in range(Time_step):
                    max1 = np.argmax(out_tmp[m,:])
                    max2 = np.argmax(target_tmp[m,:])
                    if max1 == max2:
                        correct_count += 1
                rate = correct_count/Time_step
                print(f"epoch:{epoch}, step:{step}, correctRate:{rate*100}%,loss:{loss_tmp}")

                epoch_r.append(epoch)
                step_r.append(step)
                correct_r.append(rate)
                loss_r.append(loss_tmp)
    # generating training log
    train_log = [epoch_r, step_r, correct_r, loss_r]
    train_log = pd.DataFrame(train_log).transpose()
    train_log.columns = ['epoch', "step", "correct", "loss"]
    train_log.to_csv(f'{Path}/log.csv')
    torch.save(net, f"{Path}/net.pkl")


