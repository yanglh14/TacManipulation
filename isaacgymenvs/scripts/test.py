import numpy as np
import time
import matplotlib.pyplot as plt

data = np.load('../real_controller.npy', allow_pickle=True)
cal_table = np.load('calibration_table.npy')
data = data.item()
# time_log = data['time_log']
# time_log = np.array(time_log)-time_log[0]
# control_freq =[]
# for i in range(time_log.shape[0]-1):
#     control_freq.append(1/(time_log[i+1]-time_log[i]))

tactile_log = np.array(data['tactile_log'])
tactile_pos_log = np.array(data['tactile_pos_log'])

# fig, axs = plt.subplots(4,4)
# for i in range(16):
#     axs[int(i / 4), i % 4].plot(np.array(actions_log)[:,i])
#
# plt.show()

for i in range(200):
    x = tactile_pos_log[i,:,0]
    y = tactile_pos_log[i,:,1]
    z = tactile_pos_log[i,:,2]
    tac = tactile_log[i,:]
    cal_table_ = cal_table.copy()
    cal_table_[:113] = cal_table[540:]
    cal_table_[113:] = cal_table[:540]

    tac[tac < cal_table_[:, 1]] = 0
    tac[tac > cal_table_[:, 2]] = cal_table_[tac > cal_table_[:, 2], 2]
    tac *= cal_table_[:, 0]


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=(tac) * 1000+1)

    plt.show()
