#!/home/yang/anaconda3/envs/rlgpu/bin/python
import sys
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from allegro_tactile_cal.msg import tactile_msgs

import time
import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator

from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
import torch

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

class AllegroHand():
    def __init__(self):
        self.pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        self.sub_states = rospy.Subscriber("/allegroHand_0/joint_states", JointState,self.joint_state_callback)
        self.sub_tactile = rospy.Subscriber('allegro_tactile', tactile_msgs, self.allegro_tactile_callback)

        rospy.init_node('allegro_hand', anonymous=True)
        self.rate = rospy.Rate(50)

        self.launch_rlg_hydra()
        self.player = self.runner.sim2real()

        self.env = self.player.env
        self.max_steps = 200
        self.is_determenistic = True
        self.num_obs = 82
        self.device = 'cuda:0'
        self.num_shadow_hand_dofs = 16
        self.obs_buf = torch.zeros(
            (1, self.num_obs), device=self.device, dtype=torch.float)
        self.tac_buf = torch.zeros(
            (1, 653), device=self.device, dtype=torch.float)
    def player_run(self):

        obses = self.player.env_reset(self.env)
        batch_size = 1

        for n in range(self.max_steps):
            obses = self.get_obs()
            self.action = self.player.get_actions(obses, self.is_determenistic)
            self.step(self.action)

    def joint_state_callback(self, msg):
        self.finger_jnt_pos = np.asarray(msg.position)
        self.finger_jnt_vel = np.asarray(msg.velocity)
        self.finger_jnt_force = np.asarray(msg.velocity)

    def allegro_tactile_callback(self, msg):
        self.tac_buf[:72] = np.asarray(msg.index_tip_Value)
        self.tac_buf[72:72+36] = np.asarray(msg.index_tip_Value)
        self.tac_buf[72+36:144] = np.asarray(msg.index_end_Value)
        self.tac_buf[144:144+72] = np.asarray(msg.middle_tip_Value)
        self.tac_buf[144+72:144+108] = np.asarray(msg.middle_mid_Value)
        self.tac_buf[144+108:288] = np.asarray(msg.middle_end_Value)
        self.tac_buf[288:288+72] = np.asarray(msg.ring_tip_Value)
        self.tac_buf[288+72:288+108] = np.asarray(msg.ring_mid_Value)
        self.tac_buf[288+108:288+144] = np.asarray(msg.ring_end_Value)
        self.tac_buf[432:432+72] = np.asarray(msg.thumb_tip_Value)
        self.tac_buf[432+72:432+108] = np.asarray(msg.thumb_mid_Value)
        self.tac_buf[540:653] = np.asarray(msg.palm_Value)


    def get_obs(self):
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.finger_jnt_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.env.vel_obs_scale * self.finger_jnt_vel
        self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.env.force_torque_obs_scale * self.finger_jnt_force

        obj_obs_start = 3 * self.num_shadow_hand_dofs  # 48
        self.obs_buf[:, obj_obs_start:obj_obs_start + 6] = self.env.object_pre(self.finger_jnt_pos,self.tac_buf) * 100
        self.obs_buf[:, obj_obs_start + 6:obj_obs_start + 12] = 0

        goal_obs_start = obj_obs_start + 12  # 60
        self.obs_buf[:, goal_obs_start:goal_obs_start + 6] = self.env.goal_pos

        touch_sensor_obs_start = goal_obs_start + 6  # 66

        obs_end = touch_sensor_obs_start  # 66
        # obs_total = obs_end + num_actions = 66 + 16 = 82

        self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions


    def step(self):

        joint_command = JointState()

        joint_command.header = Header()
        joint_command.header.stamp = rospy.Time.now()
        joint_command.name = ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0','joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0','joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0','joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0']
        joint_command.position = self.act
        joint_command.velocity = []
        joint_command.effort = []
        for i in range(1):
            self.pub.publish(joint_command)
            self.rate.sleep()

    @hydra.main(config_name="config", config_path="./cfg")
    def launch_rlg_hydra(self, cfg: DictConfig):

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

        # register the rl-games adapter to use inside the runner
        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
        })

        rlg_config_dict = omegaconf_to_dict(cfg.train)

        # convert CLI arguments into dictionory
        # create runner and set the settings
        self.runner = Runner(RLGPUAlgoObserver())
        self.runner.load(rlg_config_dict)
        self.runner.reset()

        # dump config dict
        experiment_dir = os.path.join('runs', cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    try:
        hand = AllegroHand()

    except rospy.ROSInterruptException:
        pass