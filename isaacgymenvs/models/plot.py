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
task_name = 'train_loss'

# loss_cnn = np.load(dir_path+'ball_cnn_'+task_name+'.npy')
# loss_gnn = np.load(dir_path+'ball_gnn_'+task_name+'.npy')
# loss_gnn_binary = np.load(dir_path+'ball_gnn_binary_'+task_name+'.npy')
# loss_mlp = np.load(dir_path+'ball_mlp_'+task_name+'.npy')
# loss_mlp_64 = np.load(dir_path+'ball_mlp_64_'+task_name+'.npy')
# loss_gnn_binary_2layer = np.load(dir_path+'ball_gnn_binary_2layer_'+task_name+'.npy')
# loss_gnn_binary_4layer = np.load(dir_path+'ball_gnn_binary_4layer_'+task_name+'.npy')
# loss_gnn_binary_5layer = np.load(dir_path+'ball_gnn_binary_5layer_'+task_name+'.npy')
# loss_gnn_binary_64channels = np.load(dir_path+'ball_gnn_binary_64channels_'+task_name+'.npy')
# loss_gnn_binary_16channels = np.load(dir_path+'ball_gnn_binary_16channels_'+task_name+'.npy')
# loss_gnn_binary_nofps = np.load(dir_path+'ball_gnn_binary_nofps_'+task_name+'.npy')

loss_gnn_binary_64channels = np.load(dir_path+'ball_gnn_binary_64channels_'+task_name+'.npy')
loss_gnn_binary_128channels = np.load(dir_path+'ball_gnn_binary_128channels_'+task_name+'.npy')
loss_gnn_binary = np.load(dir_path+'ball_gnn_binary_'+task_name+'.npy')
loss_gnn_binary_16channels = np.load(dir_path+'ball_gnn_binary_16channels_'+task_name+'.npy')
loss_gnn_binary_64channels_noself = np.load(dir_path+'ball_gnn_64channels_noself_'+task_name+'.npy')
loss_gnn_binary_64channels_nofps = np.load(dir_path+'ball_gnn_64channels_nofps_'+task_name+'.npy')

loss_mlp_64 = np.load(dir_path+'ball_mlp_'+task_name+'.npy')
loss_cnn = np.load(dir_path+'ball_cnn_'+task_name+'.npy')

loss_gnn_binary = smooth_data(loss_gnn_binary)
loss_gnn_binary_16channels = smooth_data(loss_gnn_binary_16channels)
loss_gnn_binary_64channels = smooth_data(loss_gnn_binary_64channels)
loss_gnn_binary_64channels_noself = smooth_data(loss_gnn_binary_64channels_noself)
loss_gnn_binary_64channels_nofps = smooth_data(loss_gnn_binary_64channels_nofps)

loss_gnn_binary_128channels = smooth_data(loss_gnn_binary_128channels)
loss_mlp_64 = smooth_data(loss_mlp_64)
loss_cnn = smooth_data(loss_cnn)

# Plot losses
plt.figure(figsize=(10, 8))

# plt.plot(loss_cnn, label='cnn')
# plt.plot(loss_gnn[:], label='gnn')
plt.plot(loss_gnn_binary[:], label='gnn')
# plt.plot(loss_gnn_binary_2layer[:], label='gnn_binary_2layer')
# plt.plot(loss_gnn_binary_4layer[:], label='gnn_binary_4deeper')
# plt.plot(loss_gnn_binary_5layer[:], label='gnn_binary_5deeper')
# plt.plot(loss_gnn_binary_16channels[:], label='gnn_16channels')
# plt.plot(loss_gnn_binary_64channels_noself[:], label='loss_gnn_binary_64channels_noself')
# plt.plot(loss_gnn_binary_64channels_nofps[:], label='loss_gnn_binary_64channels_nofps')

# plt.plot(loss_gnn_binary_64channels[:], label='gnn_64channels')
# plt.plot(loss_gnn_binary_128channels[:], label='gnn_128channels')
# plt.plot(loss_gnn_binary_16channels[:], label='loss_gnn_binary_16channels')
# plt.plot(loss_gnn_binary_nofps[:], label='gnn_binary_nofps')
# plt.plot(loss_mlp_64, label='loss_mlp_64')
plt.plot(loss_cnn, label='cnn')
#
plt.plot(loss_mlp_64, label='mlp')


plt.semilogy()

plt.xlabel('Epoch')
plt.ylabel('Average Loss')
# plt.grid()

plt.ylim([0,200])
plt.legend(loc=1)
# plt.title('loss')
plt.savefig('./'+task_name)
plt.show()

