import os.path

import torch
from torch.utils.data import random_split
import pickle
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 24

dir_path = 'checkpoint/'
task_name = 'val_loss'

loss_cnn = np.load(dir_path+'ball_cnn_'+task_name+'.npy')
loss_gnn = np.load(dir_path+'ball_gnn_'+task_name+'.npy')
loss_gnn_binary = np.load(dir_path+'ball_gnn_binary_'+task_name+'.npy')
loss_mlp = np.load(dir_path+'ball_mlp_'+task_name+'.npy')
loss_mlp_64 = np.load(dir_path+'ball_mlp_64_'+task_name+'.npy')

loss_gnn_binary_2layer = np.load(dir_path+'ball_gnn_binary_2layer_'+task_name+'.npy')
loss_gnn_binary_4layer = np.load(dir_path+'ball_gnn_binary_4layer_'+task_name+'.npy')
loss_gnn_binary_5layer = np.load(dir_path+'ball_gnn_binary_5layer_'+task_name+'.npy')
loss_gnn_binary_64channels = np.load(dir_path+'ball_gnn_binary_64channels_'+task_name+'.npy')
loss_gnn_binary_16channels = np.load(dir_path+'ball_gnn_binary_16channels_'+task_name+'.npy')

loss_gnn_binary_nofps = np.load(dir_path+'ball_gnn_binary_nofps_'+task_name+'.npy')

# Plot losses
plt.figure(figsize=(10, 8))
# plt.plot(loss_cnn, label='cnn')
# plt.plot(loss_gnn[:], label='gnn')
plt.plot(loss_gnn_binary[:], label='gnn_binary')
# plt.plot(loss_gnn_binary_2layer[:], label='gnn_binary_2layer')
# plt.plot(loss_gnn_binary_4layer[:], label='gnn_binary_4deeper')
# plt.plot(loss_gnn_binary_5layer[:], label='gnn_binary_5deeper')
plt.plot(loss_gnn_binary_64[:], label='loss_gnn_binary_64')

plt.plot(loss_gnn_binary_64channels[:], label='loss_gnn_binary_64channels')
plt.plot(loss_gnn_binary_16channels[:], label='loss_gnn_binary_16channels')
# plt.plot(loss_gnn_binary_nofps[:], label='gnn_binary_nofps')

# plt.plot(loss_mlp, label='mlp')
# plt.plot(loss_mlp_64, label='loss_mlp_64')

# plt.semilogy()

# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# plt.grid()
plt.ylim([0,2])
plt.legend(loc=0)
# plt.title('loss')
plt.savefig('./'+task_name)
plt.show()

