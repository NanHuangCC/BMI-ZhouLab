import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import NetModel
import pandas as pd
import time
import os

if __name__ == "__main__":
    # parameter define
    Time_step = 30
    Input_size = 2
    LR = 0.02
    WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel'  # setting work-folder
    NetDes = "test"
    LocTime = time.strftime('%Y%m%d-%H%M', time.localtime())
    Path = f'{WorkingFolder}/{LocTime}_{NetDes}'  # define cutting space video
    os.makedirs(f'{Path}')



    # use rnn model
    net = NetModel.Rnn(input_size=Input_size, outsize=1, num_layers=2, hidden_size=50)
    print(net)

    # setting net parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # optim
    loss_func = nn.MSELoss()

    h_state = None

    loss_v = []
    out_v = []
    target = []
    for step in range(500):
        start, end = step * np.pi, (step + 3) * np.pi
        steps = np.linspace(start, end, Time_step, dtype=np.float32)
        x_np = np.sin(steps)
        print(x_np)
        x1_np = np.sin(steps + 0.5)
        print(x1_np)
        x_np = np.row_stack((x_np, x1_np))
        x_np = x_np.T

        y_np = np.cos(steps)

        print(x_np.shape)

        x = torch.from_numpy(x_np[np.newaxis, :, :])
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

        print(x.shape)
        print(y.shape)

        out, h_state = net(x, h_state)
        h_state = h_state.data
        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tmp = loss.data.numpy()
        loss_v.append(loss_tmp)

        out_tmp = out.data.numpy()
        out_v.append(out_tmp)
        print(out_v)

        target_tmp = y.data.numpy()
        target.append(target_tmp)

        if step % 21 == 0:
            plt.cla()
            plt.plot(steps, y_np.flatten(), 'r-')
            plt.plot(steps, out.data.numpy().flatten(), 'b-')
            plt.pause(0.1)


    # generating training log
    loss_v = np.asarray(loss_v)
    loss_v = np.reshape(loss_v, (np.size(loss_v), 1))
    out_v = np.asarray(out_v)
    out_v = np.reshape(out_v, (np.size(loss_v), Time_step))
    target = np.asarray(target)
    target = np.reshape(target, (np.size(loss_v), Time_step))

    train_log = np.append(out_v, target, axis=1)
    train_log = np.append(train_log, loss_v, axis=1)
    # generating colnames for train log
    colname = []
    for name in ["out", "target"]:
        for i in range(1, (Time_step+1), 1):
            colname.append(name + str(i))
    colname.append("loss")

    train_log = pd.DataFrame(train_log)
    train_log.columns = colname

    train_log.to_csv(f'{Path}/train_log.csv')
    torch.save(net, f"{Path}/net.pkl")

    print(loss_v.shape)
    print(out_v.shape)
    print(target.shape)
    print(train_log.shape)



