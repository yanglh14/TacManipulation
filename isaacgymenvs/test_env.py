import isaacgym
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np
import matplotlib.pyplot as plt
from isaacgym.torch_utils import *

from utils.utils import set_np_formatting, set_seed

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import yaml
import torch
import numpy as np
import time
## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    create_rlgpu_env = get_rlgames_env_creator(
        omegaconf_to_dict(cfg.task),
        cfg.task_name,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        multi_gpu=cfg.multi_gpu,
    )

    env = create_rlgpu_env(_sim_device='cuda:0', _rl_device='cuda:0')

    actions = torch.as_tensor(np.ones([1,16])*(0),dtype=torch.float32,device='cuda:0')

    actions = unscale(actions,env.shadow_hand_dof_lower_limits, env.shadow_hand_dof_upper_limits)
    _net_cf = env.gym.acquire_net_contact_force_tensor(env.sim)
    net_cf = gymtorch.wrap_tensor(_net_cf)

    for i in range(50):
        env.step(actions)

    while True:

        for index in range(16):
            if index % 4 == 0:
                angle_list = [0, 15, 30, 15, 0, -15, -30, -15, 0]
            else:
                angle_list = [0, 15, 30, 45, 60, 45, 30, 15, 0]

            angle_obs = []

            for angle in angle_list:
                print(angle)
                actions[:] = 0
                actions[0,index] = angle * np.pi /180
                actions = unscale(actions, env.shadow_hand_dof_lower_limits, env.shadow_hand_dof_upper_limits)

                for i in range(50):
                    env.step(actions)
                    angle_obs.append(env.shadow_hand_dof_pos[0,:].cpu().detach().tolist())
            np.save('runs/joint_%d_sim'%index,np.array(angle_obs))

        break
        # env.gym.refresh_net_contact_force_tensor(env.sim)
        #
        # touch_sensor = env.net_cf[:,env.sensors_handles,:]
        # tactile = touch_sensor[0,:,2]
        # tactile_pose = env.rigid_body_states[0,env.sensors_handles,:3]
        # plot_tactile(tactile,tactile_pose)

        # contacts = env.gym.get_env_rigid_contacts(env.envs[0])
        #
        # a = contacts[contacts['body0']==199]
        # b = contacts[contacts['body1']==199]

def plot_tactile(tactile,tactile_pose):

    tac = np.abs(np.array(tactile.to('cpu')))
    tac_pose= np.array(tactile_pose.to('cpu'))

    print(tac.argmax(),tac.max())
    tac[tac<0.0005] = 0

    x = tac_pose[:, 0]
    y = tac_pose[:, 1]
    z = tac_pose[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,s=(tac) * 1000 )

    # ax = fig.add_subplot(111)
    # ax.scatter(x,y,s=(tac) * 1000 )

    plt.show()
if __name__ == "__main__":
    launch_rlg_hydra()
