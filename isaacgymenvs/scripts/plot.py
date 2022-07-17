import matplotlib.pyplot as plt
import numpy as np


def compute_gradient(data,gap=1):
    gradient = []
    for i in range(data.shape[0] - gap):
        gradient.append(data[i + gap] - data[i])

    return gradient

for i in range(1,16):
    index = "%d"%i
    joint_pos_real = np.load('real/joint_pos_'+index+'.npy')

    joint_pos_sim = np.load('sim/joint_'+index+'_sim.npy')

    joint_pos_real = joint_pos_real [:,i]
    joint_pos_sim = joint_pos_sim[:,i]

    real_gradient = compute_gradient(joint_pos_real,gap=6)
    sim_gradient = compute_gradient(joint_pos_sim)
    joint_pos_real = joint_pos_real[np.where(np.array(real_gradient) > 0.001)[0][0] - 333:]

    joint_pos_target = np.concatenate([np.zeros(1000),np.ones(1000)*np.pi*15/180,np.ones(1000)*np.pi*30/180])

    time_real = np.array(range(joint_pos_real.shape[0]))/333
    time_sim = np.array(range(joint_pos_sim.shape[0]))/50
    time_target = np.array(range(joint_pos_target.shape[0]))/1000


    fig, axs = plt.subplots(1)
    axs.plot(time_real[:999], joint_pos_real[:999],label='real')
    axs.plot(time_sim[:150], joint_pos_sim[:150],label='sim')
    axs.plot(time_target[:], joint_pos_target[:],label='target')
    axs.legend()
    plt.show()
