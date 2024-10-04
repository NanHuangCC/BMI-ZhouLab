import shap
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

# read Net need to be explain
NetFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\NetModel\20240401-1557_clusterSize4000_Neuron1000_DropOut_CNN'  # setting work-folder
netPath = f'{NetFolder}/seqnet.pkl'  # define net path
# reload net
model = torch.load(netPath)
model = model.cpu()

WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\BehavioralSet\BehavioralFeature'  # setting work-folder
Path = f'{WorkingFolder}/Full_umap_newlabels.csv'  # define cutting space video
x, y = dataProcess_BMI.get_trainSet(Path, clusterSize=4000, labelname="labels2")

x = x.cpu()
background = x[:10000]
test = x[10000:11500]
y = y.cpu()
labels = y[10000:11500].numpy()

explainer = shap.DeepExplainer(model=model, data=background)
shap_values = explainer.shap_values(test, ranked_outputs=None)

print(type(shap_values))
print(shap_values.shape)


ResultFolder = r'Z:\Project-NC-2023-A-02\1 - data\People\nan\NetExplain\20240401'
for i in range(shap_values.shape[2]):
    np.savetxt(f'{ResultFolder}/Shap{i}.csv', shap_values[:, :, i], delimiter=",")

np.savetxt(f'{ResultFolder}/Label.csv', labels, delimiter=",")



