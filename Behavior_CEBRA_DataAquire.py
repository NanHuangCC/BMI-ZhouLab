# Reloading data and CEBRA_model
import pandas as pd
import cebra.models
import matplotlib
import numpy as np
import cv2 as cv
# Read data from folder for CEBRA
WorkingFolder = r'D:\project\BCI\data\data_30\20230520FOV\Behavior\CEBRA'   # path of behavior_data
Behavior_data = cebra.load_data(file=f'{WorkingFolder}\Full_Inf.h5', key="Behavior_Inf")
# Reload CEBRA_model
cebra_model = cebra.CEBRA.load(f'{WorkingFolder}\cebra_model.pt')

# Model evaluation
#  Parameter setting
out_dim = 8
timesteps = len(Behavior_data)

# get embedding
embedding = cebra_model.transform(Behavior_data)
assert(embedding.shape == (timesteps, out_dim))

pd.DataFrame(embedding).to_csv(f'{WorkingFolder}\CEBRA_embedding.csv')
