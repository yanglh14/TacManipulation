import matplotlib.pyplot as plt
import numpy as np


for i in range(12,16):
    index = "%d"%i
    print(index)
    joint_pos_real = np.load('real/joint_real_'+index+'.npy')

    joint_pos_sim = np.load('sim/joint_sim_'+index+'.npy')

    joint_pos_real = joint_pos_real [:,i]
    joint_pos_sim = joint_pos_sim[:,i]

    joint_pos_real[:] = joint_pos_real[:] - joint_pos_real[0]
    if i == 12 or i == 14:
        joint_pos_real[:] += 20*np.pi/180

    if i == 12 or i ==  14:
        angle_list = [20, 30, 45, 60, 75, 60, 45, 30, 20]
    else:
        if i % 4 == 0:
            angle_list = [0, 15, 25, 15, 0, -15, -25, -15, 0]
        else:
            angle_list = [0, 15, 30, 45, 60, 45, 30, 15, 0]

    joint_pos_target = np.concatenate([np.ones(50)*np.pi*angle_list[0]/180,np.ones(50)*np.pi*angle_list[1]/180,np.ones(50)*np.pi*angle_list[2]/180,
                                       np.ones(50)*np.pi*angle_list[3]/180,np.ones(50)*np.pi*angle_list[4]/180,np.ones(50)*np.pi*angle_list[5]/180,
                                       np.ones(50)*np.pi*angle_list[6]/180,np.ones(50)*np.pi*angle_list[7]/180,np.ones(50)*np.pi*angle_list[8]/180,
                                       ])

    fig, axs = plt.subplots(1)
    axs.plot(joint_pos_real[:],label='real')
    axs.plot(joint_pos_sim[:],label='sim')
    axs.plot(joint_pos_target[:],label='target')
    axs.legend()
    plt.show()

fig, axs = plt.subplots(1)
axs.plot(train_loss,label='train')
axs.plot(val_loss,label='val')
axs.legend()
plt.show()
