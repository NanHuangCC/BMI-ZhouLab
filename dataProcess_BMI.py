import numpy as np
import pandas as pd
import numpy
import re
import torch
from sklearn.preprocessing import MinMaxScaler

def get_labels(file_path, start, end):
    clusters = pd.read_csv(file_path)  # read csv
    clusters.rename(columns={'Unnamed: 0': 'rank'}, inplace=True)
    clusters["time"] = ((numpy.arange(0, len(clusters.index), 1)) / 30) + 2

    clusters = clusters.loc[(clusters["labels"] >= 0)]
    clusters.index = range(clusters.shape[0])
    clusters = clusters.loc[start:end]

    pattern = "_"
    matched_list = []
    for item in clusters.columns:
        if re.search(pattern, item):
            matched_list.append(item)
    targets = clusters["labels"]
    targets = targets.values
    matched_list.append("velocity")

    input_data = clusters[matched_list]
    input_data = input_data.values

    # trans labels to 0-1 sequence
    targets_seq = numpy.zeros((clusters.shape[0], (max(targets) + 1)))
    for i in range(clusters.shape[0]):
        targets_seq[i, targets[i]] = 1

    x = torch.from_numpy(input_data[np.newaxis, :, :]).type(torch.FloatTensor)
    y = torch.from_numpy(targets_seq[np.newaxis, :, :]).type(torch.FloatTensor)

    return x, y


# get train_set
def get_trainSet(file_path, clusterSize, labelname):
    labelname = labelname
    minSetSize = clusterSize
    file_path = file_path
    clusters = pd.read_csv(file_path)  # read csv
    clusters.rename(columns={'Unnamed: 0': 'rank'}, inplace=True)
    clusters = clusters.loc[(clusters[labelname] >= 0)]

    # get cluster list
    labels = clusters[labelname].tolist()
    count_set = set(labels)
    # conbine TrainSet
    TrainSet = clusters.head(1)
    clusters = clusters.drop(index=clusters.index[0])
    for cluster in count_set:
        clusters_tmp = clusters.loc[(clusters[labelname] == cluster)]
        if len(clusters_tmp.index) >= minSetSize:
            clusters_tmp = clusters_tmp.sample(n=minSetSize)
        else:
            clusters_tmp = clusters_tmp
        TrainSet = pd.concat([TrainSet, clusters_tmp])
    TrainSet = TrainSet.drop(index=TrainSet.index[0])
    # shuffle TrainSet
    TrainSet = TrainSet.sample(n=(len(TrainSet.index)))

    pattern = "_"
    matched_list = []
    for item in TrainSet.columns:
        if re.search(pattern, item):
            matched_list.append(item)
    matched_list.append("velocity")
    input_data = TrainSet[matched_list]
    input_data = input_data.values

    targets = TrainSet[labelname]
    targets = targets.values
    # trans labels to 0-1 sequence
    targets_seq = numpy.zeros((TrainSet.shape[0], (max(targets) + 1)))
    for i in range(TrainSet.shape[0]):
        targets_seq[i, targets[i]] = 1

    x = torch.from_numpy(input_data).type(torch.FloatTensor)
    y = torch.from_numpy(targets_seq).type(torch.FloatTensor)
    return x, y


# get train_set
def pd2tensor(file_path):
    file_path = file_path
    clusters = pd.read_csv(file_path)  # read csv
    pattern = "_"
    matched_list = []
    for item in clusters.columns:
        if re.search(pattern, item):
            matched_list.append(item)
    matched_list.append("velocity")
    input_data = clusters[matched_list]
    input_data = input_data.values
    x = torch.from_numpy(input_data).type(torch.FloatTensor)
    return x


# combined data transfer
def combined2tensor(data, inputTag="neu", targerTag="out", Time_step=60, agg_row=3, neuLim=False, neucounts=20):
    # get input matrix
    # basic function
    data = data

    def grep_colum(pattern, data):
        pattern = pattern
        data = data
        matched_list = []
        for item in data.columns:
            if re.search(pattern, item):
                matched_list.append(item)
        dataG = data[matched_list]
        return dataG

    def split_data(data, time_step=30, agg_row=3):
        dataX = []
        cutL = int(time_step / agg_row)
        for i in range(len(data) - cutL):
            dataX.append(data[i:i + cutL])
        dataX = np.array(dataX).reshape(len(dataX), cutL, -1)
        return dataX

    # split combined data based on Tag
    Set1 = grep_colum(pattern=inputTag, data=data)
    Set2 = grep_colum(pattern=targerTag, data=data)

    # add time dim to a frame
    scaler = MinMaxScaler()
    train_use = Set1[Set1.columns[0]].values
    agg_count = int(len(train_use) / agg_row)
    train_use_new = np.zeros(agg_count)
    for i in range(agg_count):
        start_pos = i * agg_row
        end_pos = (i + 1) * agg_row
        train_use_new[i] = sum(train_use[start_pos:end_pos])
    train_use_new = scaler.fit_transform(train_use_new.reshape(-1, 1))
    Set1S = split_data(data=train_use_new, time_step=Time_step, agg_row=agg_row)

    if neuLim:
        neucounts = neucounts
        for neu in Set1.columns[1:neucounts]:
            train_use = Set1[neu].values
            train_use_new = np.zeros(agg_count)
            for i in range(agg_count):
                start_pos = i * agg_row
                end_pos = (i + 1) * agg_row
                train_use_new[i] = sum(train_use[start_pos:end_pos])
            train_use_new = scaler.fit_transform(train_use_new.reshape(-1, 1))
            Set1S_tmp = split_data(train_use_new, Time_step, agg_row=agg_row)
            Set1S = np.concatenate((Set1S, Set1S_tmp), axis=2)
    else:
        for neu in Set1.columns[1:]:
            train_use = Set1[neu].values
            train_use_new = np.zeros(agg_count)
            for i in range(agg_count):
                start_pos = i * agg_row
                end_pos = (i + 1) * agg_row
                train_use_new[i] = sum(train_use[start_pos:end_pos])
            train_use_new = scaler.fit_transform(train_use_new.reshape(-1, 1))
            Set1S_tmp = split_data(train_use_new, Time_step, agg_row=agg_row)
            Set1S = np.concatenate((Set1S, Set1S_tmp), axis=2)

    # generate targets
    target_raw = np.array(Set2)[Time_step:]
    target_new = np.zeros((int(len(target_raw) / agg_row), np.shape(target_raw)[1]))
    for i in range(int(len(target_raw) / agg_row)):
        start_pos = i * agg_row
        end_pos = (i + 1) * agg_row
        agg = target_raw[start_pos:end_pos, :]
        target_new[i:] = agg.mean(axis=0)
    target = np.zeros(np.shape(target_new))
    for i in range(0, len(target_new), 1):
        max1 = np.argmax(target_new[i, :])
        target[i, max1] = 1
    target = target.astype(np.float64)

    # get final value
    x = torch.Tensor(Set1S)
    y = torch.Tensor(target)
    return x, y

def ReadBehNeu(Beh_path, Neu_path, TauT = 3, FramePerS = 5):
    # read data
    Beh_data = pd.read_csv(Beh_path)
    Neu_data = pd.read_csv(Neu_path)
    Beh_data.drop([Beh_data.columns[0]], axis=1, inplace=True)
    Neu_data.drop([Neu_data.columns[0]], axis=1, inplace=True)

    # trans df to numpy & keep targets
    Beh_data = Beh_data.values
    Neu_data = Neu_data.values

    # Construct dataset
    TauFrame = TauT * FramePerS

    # get prediction target & Input Xt
    Xt = Neu_data[TauFrame:(Beh_data.shape[0] - TauFrame), :]  # Xt range
    Yt = Neu_data[(TauFrame + 1):(Beh_data.shape[0] - TauFrame + 1), :]

    # get Behavioral input
    # Set void data Set
    SenSize = Xt.shape[0] * TauFrame * Beh_data.shape[1]
    SensoryFeedback = np.zeros(SenSize)
    SensoryFeedback = np.reshape(SensoryFeedback, (Xt.shape[0], 1, TauFrame, Beh_data.shape[1]))
    MovementFeedforward = np.zeros(SenSize)
    MovementFeedforward = np.reshape(MovementFeedforward, (Xt.shape[0], 1, TauFrame, Beh_data.shape[1]))

    print(SensoryFeedback.shape)
    print(range(Xt.shape[0]))
    # fill data into data set
    for i in range(Xt.shape[0]):
        SensoryFeedback[i, 0, :, :] = Beh_data[i:(i + TauFrame), :]
        MovementFeedforward[i, 0, :, :] = Beh_data[(i + TauFrame + 1):(i + TauFrame * 2 + 1), :]

    Ust = torch.from_numpy(SensoryFeedback).type(torch.FloatTensor)
    Umt = torch.from_numpy(MovementFeedforward).type(torch.FloatTensor)
    Xt = torch.from_numpy(Xt).type(torch.FloatTensor)
    Yt = torch.from_numpy(Yt).type(torch.FloatTensor)
    return Ust, Umt, Xt, Yt
