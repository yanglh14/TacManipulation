import os.path

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from gnn import *
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library


def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[1], dst[1]], [src[2], dst[2]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 1], pos[:, 2], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.xticks([])
    plt.yticks([])

    plt.show()


# Check if the GPU is available
device = torch.device("cpu")
print(f'Selected device: {device}')

model_dir = './models'
if_model = True
save_dir = '../tac_data/'

task_name = 'final_multi'
object_name = 'tac_data_multi'


with open(save_dir+object_name+'.pkl','rb') as f:
    d = pickle.load(f)
tac = abs(torch.tensor(d['tactile'],dtype=torch.float32, device=device)[:,:,2])
pos = torch.tensor(d['tac_pose'],dtype=torch.float32, device=device)
y = torch.tensor(d['class'],dtype=torch.long, device=device)

tac /= tac.max(1,keepdim=True)[0]
tac = tac.view(10000,-1,1)

tactile_dataset = []
for i in range(3000,4000):

    data = Data(x=tac[i,tac[i,:,0]!=0,:],pos=pos[i,tac[i,:,0]!=0,:],y=y[i])
    tactile_dataset.append(data)

data = tactile_dataset[5]
visualize_points(data.pos, data.edge_index)

from torch_cluster import knn_graph

data.edge_index = knn_graph(data.pos, k=4)
print(data.edge_index.shape)
visualize_points(data.pos, edge_index=data.edge_index)