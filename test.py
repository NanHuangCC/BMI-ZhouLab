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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set analysis session

T1 = np.arange(0, 3000)
T2 = np.arange(3600, 6600)
T3 = np.arange(7200, 10200)
T4 = np.arange(10800, 13800)
BGrange = np.concatenate((T1,T2,T3,T4), axis=0)



print(BGrange)