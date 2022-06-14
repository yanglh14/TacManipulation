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

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 24

dir_path = 'data/'
task_name = 'accuracy'

loss_cnn = np.load(dir_path+'final_2_classifier_'+task_name+'.npy')
loss_gnn = np.load(dir_path+'final_multi_2_classifier_'+task_name+'.npy')
loss_lstm = np.load(dir_path+'final_dynamic_2_classifier_'+task_name+'.npy')

# Plot losses
plt.figure(figsize=(10, 8))
plt.plot(loss_cnn, label='2D Static')
plt.plot(loss_gnn[:30], label='3D Static')
plt.plot(loss_lstm, label='Dynamic')

plt.xlabel('Epoch')
plt.ylabel('Average Loss')
# plt.grid()
plt.legend()
# plt.title('loss')
plt.savefig('../Pictures/'+task_name)
plt.show()

