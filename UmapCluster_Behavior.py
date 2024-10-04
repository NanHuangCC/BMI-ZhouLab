'''
This file for data process form matlab
(online data process use python too)
'''
# import kits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re


# import cluster kits
from sklearn.datasets import fetch_openml
import hdbscan

WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\0data\BehavioralSet\BehavioralFeature'
ReadFile = f'{WorkingFolder}/Full_Inf.csv'
SaveFile = f'{WorkingFolder}/Full_umap.csv'

# Read data from folder
Full_Inf = pd.read_csv(ReadFile, sep=",")


pattern = "_"
matched_list = []
for item in Full_Inf.columns:
    if re.search(pattern, item):
        matched_list.append(item)
matched_list.append("velocity")
data = Full_Inf[matched_list]

scaled_data = StandardScaler().fit_transform(data)

# UMAP reduce dims
reducer = umap.UMAP(n_components=6, min_dist=0.00, n_neighbors=900, metric="cosine")
embedding = reducer.fit_transform(scaled_data)

# clustering use UMAP
labels = hdbscan.HDBSCAN(
    min_cluster_size=750,
    min_samples=3,
).fit_predict(embedding)

clustered = (labels >= 0)


# plot use labels
plt.scatter(
    embedding[clustered, 4],
    embedding[clustered, 5],
    c=labels[clustered],
    s=0.1,
    cmap='turbo'
)
plt.colorbar()
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the behavior dataset', fontsize=24)
plt.show()

# plot use labels (3D)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(embedding[clustered, 0], embedding[clustered, 4], embedding[clustered, 5],
             c=labels[clustered], s=0.1, cmap='turbo')
plt.show()

# save as csv
Full_Inf["labels"] = labels
Full_Inf["umap1"] = embedding[:, 0]
Full_Inf["umap2"] = embedding[:, 1]
Full_Inf["umap3"] = embedding[:, 2]
Full_Inf["umap4"] = embedding[:, 3]
Full_Inf["umap5"] = embedding[:, 4]
Full_Inf["umap6"] = embedding[:, 5]
Full_Inf.to_csv(SaveFile)
