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

plt.style.use('seaborn-darkgrid')

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 36

dir_path = 'checkpoint/'
task_name = 'train_loss'

data_list = ['ball_mlp_','ball_cnn_','ball_gcn_32_','ball_gnn_binary_']
# data_list = ['ball_gnn_binary_','ball_gnn_binary_16channels_','ball_gnn_binary_64channels_','ball_gnn_binary_128channels_']

label = ['32channels','16channels','64channels','128channels']
line_style = ['-','--','-.',':']
# Plot losses
plt.figure(figsize=(20,16))


for i,data in enumerate(data_list):
    loss = np.load(dir_path + data + task_name + '.npy')
    loss = smooth_data(loss)
    print(data,loss[-1])
    plt.plot(loss, label=label[i],linestyle =line_style[i],linewidth=6 )

# plt.semilogy()

# plt.title('Object Prediction Results')
# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
plt.ylim([0,2])
plt.legend(loc=0,edgecolor='blue', frameon = True)
plt.savefig('./'+task_name)
plt.show()


