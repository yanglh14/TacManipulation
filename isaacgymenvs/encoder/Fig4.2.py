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
task_name = 'accuracy'

pro_final = np.load(dir_path+'pro_final_2.npy')
pro_final_dynamic = np.load(dir_path+'pro_final_dynamic_2.npy')
pro_final_multi = np.load(dir_path+'pro_final_multi_2.npy')


x = np.arange(10,dtype=int)  # the label locations
width = 0.2  # the width of the bars

for i in range(10):
    fig, ax = plt.subplots()

    final = pro_final[i]
    final_dynamic = pro_final_dynamic[i]
    final_multi = pro_final_multi[i]

    final -= final.min()
    final_dynamic -= final_dynamic.min()
    final_multi -= final_multi.min()

    final /= final.sum() *0.01
    final_dynamic /= final_dynamic.sum() *0.01
    final_multi /= final_multi.sum() *0.01

    rects1 = ax.bar(x - width, final, width, label='2D Static')
    rects2 = ax.bar(x , final_multi, width, label='3D Static')
    rects3 = ax.bar(x + width, final_dynamic, width, label='Dynamic')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    # ax.bar_label(rects3, padding=3)

    fig.tight_layout()
    plt.savefig(f'../Pictures/pro_{i}' )
    # plt.show()



