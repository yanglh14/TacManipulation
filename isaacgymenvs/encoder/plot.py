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
        data_list.append(np.mean(data[(i) * 200+50:(i+1) * 200]))

    return(data_list)

plt.style.use('seaborn-darkgrid')

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 36

dir_path = 'checkpoint_cube/'
task_name = 'val_loss'

data_list = ['ball_mlp_32_','ball_cnn_32_','ball_gcn_32_','ball_gnn_32_']
# data_list = ['ball_gnn_binary_','ball_gnn_binary_16channels_','ball_gnn_binary_64channels_','ball_gnn_binary_128channels_']
# data_list = ['ball_mlp_','ball_cnn_','ball_gcn_','ball_gnn_']
# data_list = ['mlp_','cnn_','gcn_','gnn_']
# data_list = ['ball_mlp_32_','ball_gnn_32_']

label = ['mlp','cnn','gcn','gnn']
line_style = ['-','--','-.',':']
# label = ['mlp','gnn']
# line_style = ['-',':']
color_list = ['blue','green','orange','red']

iteration_num = [2,2,2,2]

# Plot losses
fig = plt.figure(figsize=(20,16))
ax = fig.add_subplot(1,1,1)

for i,data in enumerate(data_list):
    loss_list = []

    for j in range(iteration_num[i]):
        loss = np.load(dir_path + data + '%d_'%(j+1)+ task_name + '.npy')
        # loss = smooth_data(loss)
        print(data,loss[-1])
        loss_list.append(loss)

    loss_list = np.array(loss_list)

    loss_avg = np.mean(loss_list, axis=0)
    loss_std = np.std(loss_list, axis=0)
    r1 = list(map(lambda x: x[0] - x[1], zip(loss_avg, loss_std)))
    r2 = list(map(lambda x: x[0] + x[1], zip(loss_avg, loss_std)))

    ax.plot(list(range(30)),loss_avg, label=label[i], color = color_list[i],linestyle =line_style[i],linewidth=6 )
    ax.fill_between(list(range(30)), r1, r2, color = color_list[i], alpha=0.2)

plt.semilogy()

# plt.title('Object Prediction Results')
# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
plt.ylim([0,100])
plt.legend(loc=0,edgecolor='blue', frameon = True)
# plt.savefig('./'+task_name+'_ball')
plt.show()


