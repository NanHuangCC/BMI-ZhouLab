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

# import cluster kits
from sklearn.datasets import fetch_openml
import hdbscan

WorkingFolder = r'Z:\- BMI Updates -\0data\test\BeahaviorCluster\BehaviorFeature'
ReadFile = f'{WorkingFolder}/Full_Inf.csv'
SaveFile = f'{WorkingFolder}/Full_umap.csv'

# Read data from folder
Full_Inf = pd.read_csv(ReadFile, sep=",")
Full_Inf = Full_Inf.drop(Full_Inf.columns[[0]], axis=1)

scaled_data = StandardScaler().fit_transform(Full_Inf)

# UMAP reduce dims
reducer = umap.UMAP(n_components=3, min_dist=0.0, n_neighbors=900, metric="euclidean")
embedding = reducer.fit_transform(scaled_data)

# clustering use UMAP
labels = hdbscan.HDBSCAN(
    min_cluster_size=600,
    min_samples=4,
).fit_predict(embedding)

clustered = (labels >= 0)


# plot use labels
plt.scatter(
    embedding[clustered, 0],
    embedding[clustered, 1],
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
ax.scatter3D(embedding[clustered, 0], embedding[clustered, 1], embedding[clustered, 2],
             c=labels[clustered], s=0.1, cmap='turbo')
plt.show()

# save as csv
Full_Inf["labels"] = labels
Full_Inf["umap1"] = embedding[:, 0]
Full_Inf["umap2"] = embedding[:, 1]
Full_Inf["umap3"] = embedding[:, 2]
Full_Inf.to_csv(SaveFile)
