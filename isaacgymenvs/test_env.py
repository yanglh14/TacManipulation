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

    actions = torch.as_tensor(np.zeros([1,16]),dtype=torch.long)
    _net_cf = env.gym.acquire_net_contact_force_tensor(env.sim)
    net_cf = gymtorch.wrap_tensor(_net_cf)

    while True:
        env.step(actions)

        # index_start = env.gym.find_actor_rigid_body_index(env.envs[0], 0, 'touch_111_1_1', gymapi.DOMAIN_SIM)
        # index_end = env.gym.find_actor_rigid_body_index(env.envs[0], 0, 'touch_111_7_12', gymapi.DOMAIN_SIM)
        # sensors_handles = range(index_start, index_end+1)
        #
        # env.gym.refresh_net_contact_force_tensor(env.sim)
        #
        # touch_sensor = env.net_cf[:,env.sensors_handles,:]
        # tactile = touch_sensor[0,:,2]
        # tactile_pose = env.rigid_body_states[0,env.sensors_handles,:3]
        # plot_tactile(tactile,tactile_pose)
        # print(net_cf)


def plot_tactile(tactile,tactile_pose):

    tac = np.abs(np.array(tactile.to('cpu')))
    tac_pose= np.array(tactile_pose.to('cpu'))

    x = tac_pose[:, 0]
    y = tac_pose[:, 1]
    z = tac_pose[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)

    # ax.scatter(x, y, z, s=(tac) * 50 + 0.2)
    ax.scatter(x,y,z,s=(tac) * 1000 + 1)

    plt.show()
if __name__ == "__main__":
    launch_rlg_hydra()
