import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch


class AllegroHandNotask():

    def __init__(self):

        self.gym = gymapi.acquire_gym()
        self.device = 'cuda:0'

        self.create_sim()
        self.if_viewer = True
        if self.if_viewer:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)

            cam_pos = gymapi.Vec3(0.3, 0, 1)
            cam_target = gymapi.Vec3(0, 0, 0.7)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state_target = self.dof_state.clone()

        self.sensor_handles = self.get_sensor_handles()

    def create_sim(self):

        sim_params = gymapi.SimParams()

        # set common parameters
        sim_params.dt = 1 / 50
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # sim_params.use_gpu_pipeline = True

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1000.0
        sim_params.physx.default_buffer_size_multiplier = 10.0

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        self._create_ground_plane()
        self._create_envs(1, 0.75, 1)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        shadow_hand_asset_file = "tactile/allegro_hand/allegro_hand.xml"

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        # load finger assets
        finger1_asset_file = "tactile/allegro_hand/allegro_finger1.xml"
        finger1_asset = self.gym.load_asset(self.sim, asset_root, finger1_asset_file, asset_options)
        finger2_asset_file = "tactile/allegro_hand/allegro_finger2.xml"
        finger2_asset = self.gym.load_asset(self.sim, asset_root, finger2_asset_file, asset_options)
        finger3_asset_file = "tactile/allegro_hand/allegro_finger3.xml"
        finger3_asset = self.gym.load_asset(self.sim, asset_root, finger3_asset_file, asset_options)
        finger4_asset_file = "tactile/allegro_hand/allegro_finger4.xml"
        finger4_asset = self.gym.load_asset(self.sim, asset_root, finger4_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset) + self.gym.get_asset_rigid_body_count(finger1_asset) + self.gym.get_asset_rigid_body_count(finger2_asset) + self.gym.get_asset_rigid_body_count(finger3_asset) + self.gym.get_asset_rigid_body_count(finger4_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset) + self.gym.get_asset_rigid_shape_count(finger1_asset) + self.gym.get_asset_rigid_shape_count(finger2_asset) + self.gym.get_asset_rigid_shape_count(finger3_asset) + self.gym.get_asset_rigid_shape_count(finger4_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset) + self.gym.get_asset_dof_count(finger1_asset) + self.gym.get_asset_dof_count(finger2_asset) + self.gym.get_asset_dof_count(finger3_asset) + self.gym.get_asset_dof_count(finger4_asset)
        print("Num dofs: ", self.num_shadow_hand_dofs)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs #self.gym.get_asset_actuator_count(shadow_hand_asset)

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        # set shadow_hand dof properties
        shadow_hand_dof_props = np.concatenate((self.gym.get_asset_dof_properties(finger1_asset),self.gym.get_asset_dof_properties(finger2_asset),self.gym.get_asset_dof_properties(finger3_asset),self.gym.get_asset_dof_properties(finger4_asset)))

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

            print("Max effort: ", shadow_hand_dof_props['effort'][i])
            shadow_hand_dof_props['effort'][i] = 0.5
            shadow_hand_dof_props['stiffness'][i] = 3
            shadow_hand_dof_props['damping'][i] = 0.1
            shadow_hand_dof_props['friction'][i] = 0.01
            shadow_hand_dof_props['armature'][i] = 0.001
        data = np.load('../scripts/joint_sim2real.npy')
        shadow_hand_dof_props['stiffness'] = data[0]
        shadow_hand_dof_props['damping'] = data[1]

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, 2))
        shadow_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi/12)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies
        max_agg_shapes = self.num_shadow_hand_shapes

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        for i in range(1):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 0, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            # add finger1
            finger1_actor = self.gym.create_actor(env_ptr, finger1_asset, shadow_hand_start_pose, "finger1", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger1_actor, shadow_hand_dof_props[:4])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger1_actor)


            # add finger2
            finger2_actor = self.gym.create_actor(env_ptr, finger2_asset, shadow_hand_start_pose, "finger2", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger2_actor, shadow_hand_dof_props[4:8])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger2_actor)

            # add finger3
            finger3_actor = self.gym.create_actor(env_ptr, finger3_asset, shadow_hand_start_pose, "finger3", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger3_actor, shadow_hand_dof_props[8:12])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger3_actor)

            # add finger4
            finger4_actor = self.gym.create_actor(env_ptr, finger4_asset, shadow_hand_start_pose, "finger4", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger4_actor, shadow_hand_dof_props[12:16])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger4_actor)


            hand_idx = self.gym.get_actor_index(env_ptr, finger1_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)


            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

    def get_sensor_handles(self):

        sensors_handles = np.array([])
        for i in range(12):
            if i ==0:
                index_start = self.gym.find_actor_rigid_body_index(self.envs[0], 0, 'touch_111_1_1', gymapi.DOMAIN_SIM)
                index_end = self.gym.find_actor_rigid_body_index(self.envs[0], 0, 'touch_111_7_12', gymapi.DOMAIN_SIM)
                sensors_handles = np.concatenate((sensors_handles,np.array(range(index_start, index_end+1))))
            else:
                j = (i-1)//3 +1
                if (i-1)%3 == 0:
                    index_start = self.gym.find_actor_rigid_body_index(self.envs[0], j, 'touch_%d_1_1'%(i-1), gymapi.DOMAIN_SIM)
                    index_end = self.gym.find_actor_rigid_body_index(self.envs[0], j, 'touch_%d_6_12'%(i-1), gymapi.DOMAIN_SIM)
                    sensors_handles = np.concatenate((sensors_handles,np.array(range(index_start, index_end+1))))
                else:
                    index_start = self.gym.find_actor_rigid_body_index(self.envs[0], j, 'touch_%d_1_1'%(i-1), gymapi.DOMAIN_SIM)
                    index_end = self.gym.find_actor_rigid_body_index(self.envs[0], j, 'touch_%d_6_6'%(i-1), gymapi.DOMAIN_SIM)
                    sensors_handles = np.concatenate((sensors_handles,np.array(range(index_start, index_end+1))))

        return sensors_handles

    def step(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # update the viewer
        if self.if_viewer:
            self.gym.step_graphics(self.sim);
            self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        return self.rigid_body_states[0,self.sensor_handles,:3]

    def sim2real(self,act):

        self.dof_state_target[:,0] = torch.tensor(act,dtype=torch.float32,device=self.device)

        hand_indices = torch.tensor([1,2,3,4], dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state_target),
                                                gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # update the viewer
        if self.if_viewer:
            self.gym.step_graphics(self.sim);
            self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.sensor_pos = self.rigid_body_states[0,self.sensor_handles,:3]
        return self.sensor_pos

if __name__ == '__main__':
    hand = AllegroHandNotask()
    act = np.array([0,0,0,0,
           0,0,0,0,
           0,0,0,0,
           0,0,0,0,],dtype=np.float32)

    for i in range(100):
        act[1] = i* np.pi/200
        rigid_body_states = hand.sim2real(act)