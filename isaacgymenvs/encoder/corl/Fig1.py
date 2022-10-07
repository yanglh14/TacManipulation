import os.path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


save_dir = '../tac_data/'
object_name = '010_potted_meat_can_dynamic'

with open(save_dir+object_name+'.pkl','rb') as f:
    d = pickle.load(f)

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 24

font = {'family': 'Times New Roman',
        'size': 20,
        }

def plot_tactile_heatmap(tactile,tactile_pose,object_name=None):

    tac = np.abs(tactile)
    tac_pose= tactile_pose

    u = tac[:,0].reshape(15,15)
    v = tac[:,1].reshape(15,15)
    w = tac[:,2].reshape(15,15)
    if np.sum(w)>0:

        fig = plt.figure(figsize=(8, 8),)
        ax = fig.add_subplot(111)

        im = ax.imshow(w, cmap=plt.cm.hot_r)

        plt.xticks([])
        plt.yticks([])

        fig.colorbar(im,ticks=[])

        if object_name != None:
            plt.savefig(os.path.abspath(os.path.join(os.getcwd(),'..')) + "/figures/"+object_name+".png")
            plt.close(fig)
        plt.show()

tactile = d['tactile'][0, 9]
tactile_pose = d['tac_pose'][0]

plot_tactile_heatmap(tactile,tactile_pose)