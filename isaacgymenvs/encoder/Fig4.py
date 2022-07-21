import os.path

import torch
from torch.utils.data import random_split
import pickle
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 24

dir_path = 'data/'
task_name = 'train_loss'

loss_cnn = np.load(dir_path+'final_2_classifier_'+task_name+'.npy')
loss_gnn = np.load(dir_path+'final_multi_2_classifier_'+task_name+'.npy')
loss_lstm = np.load(dir_path+'final_dynamic_2_classifier_'+task_name+'.npy')

# Plot losses
plt.figure(figsize=(10, 8))
plt.plot(loss_cnn, label='2D Static')
plt.plot(loss_gnn[:30], label='3D Static')
plt.plot(loss_lstm, label='Dynamic')

# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# plt.grid()
plt.ylim([0,2])
plt.legend(loc=0)
# plt.title('loss')
plt.savefig('../figures/'+task_name)
plt.show()

