import os.path

import torch
from torch.utils.data import random_split
import pickle
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

def smooth_data(data):
    data_list = []
    for i in range(30):
        data_list.append(np.mean(data[(i) * 20:(i+1) * 20]))

    return(data_list)

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 24

dir_path = 'checkpoint/'
task_name = 'val_loss'

data_list = ['ball_mlp_','ball_gnn_binary_64channels_','ball_gnn_binary_']

# Plot losses
plt.figure(figsize=(10, 8))

for data in data_list:
    loss = np.load(dir_path + data + task_name + '.npy')
    loss = smooth_data(loss)
    print(data,loss[-1])
    plt.plot(loss[:], label=data)

# plt.semilogy()

# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# plt.grid()
plt.ylim([0,2])
plt.legend(loc=0)
# plt.title('loss')
plt.savefig('./'+task_name)
plt.show()


