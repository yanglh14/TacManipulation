import os.path
import pickle
import time

from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np
import matplotlib.pyplot as plt
import torch
from isaacgym.torch_utils import *

gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 50
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.use_gpu_pipeline = True

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
index = 2
device = f'cuda:{index}'
sim = gym.create_sim(index, index, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

#### load desk asset
asset_root = "../../assets"
asset_file = "tactile/CoRL2022/corl2022_multi1.xml"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
asset1 = gym.load_asset(sim, asset_root, asset_file, asset_options)

asset_options.disable_gravity = True
asset_file = "tactile/CoRL2022/corl2022_multi2.xml"
asset2 = gym.load_asset(sim, asset_root, asset_file, asset_options)
asset_file = "tactile/CoRL2022/corl2022_multi3.xml"
asset3 = gym.load_asset(sim, asset_root, asset_file, asset_options)

#### load object asset
if_viewer = False
sim_length = 200
num_iter = 1000
noise_scale = 1
object_name = 'cylinder_big'
asset_root = "../../assets"
asset_file = "tactile/objects/ycb/"+object_name+"/"+object_name+ ".urdf"
asset_file = "tactile/objects/"+ object_name +".urdf"
object_height = 0.15
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01

asset4 = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

def run_sim(if_viewer, sim_length, num_iter, noise_scale):
    # create and populate the environments
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        gym.begin_aggregate(env, 1000, 1000, True)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.1, 0.1, 0.1)
        actor_handle0 = gym.create_actor(env, asset1, pose, "Desk", i, 1)
        pose.p = gymapi.Vec3(0.1, 0.0, 0.20)
        actor_handle1 = gym.create_actor(env, asset2, pose, "Desk2", i, 1)
        pose.p = gymapi.Vec3(0.1, 0.20, 0.20)
        actor_handle2 = gym.create_actor(env, asset3, pose, "Desk3", i, 1)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.1, 0.1, object_height)
        actor_handle3 = gym.create_actor(env, asset4, pose, "Object", i, 0)

        gym.end_aggregate(env)

    if if_viewer:
        cam_props = gymapi.CameraProperties()
        viewer = gym.create_viewer(sim, cam_props)
        cam_pos = gymapi.Vec3(0.3, 0.3, 0.3)
        cam_target = gymapi.Vec3(0, 0, 0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    gym.prepare_sim(sim)

    f,p = [],[]
    step, i = 0,0
    tactile_list, tac_pose_list, object_pos_list = [],[],[]

    # while not gym.query_viewer_has_closed(viewer):
    t = time.time()
    while i < num_iter:

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        step += 1

        # update the viewer
        if if_viewer:
            gym.step_graphics(sim);
            gym.draw_viewer(viewer, sim, True)
        # refresh
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)

        dof_tensor = gym.acquire_dof_state_tensor(sim)
        dof_state = gymtorch.wrap_tensor(dof_tensor)
        # print(step,dof_state[:,0])

        cur_targets = torch.ones((1, 2), dtype=torch.float, device=f'cuda:{index}') * (0)
        if step > 50:
            cur_targets = torch.ones((1, 2), dtype=torch.float, device=f'cuda:{index}')* (0.05)
            # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(cur_targets))
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(cur_targets))

        # get pose and tactile
        _net_cf = gym.acquire_net_contact_force_tensor(sim)
        net_cf = gymtorch.wrap_tensor(_net_cf).view(1, -1, 3)
        tactile1 = net_cf[0,3:3+225,:3]
        tactile2 = net_cf[0,3+3+225:3+225+3+225,:3]
        tactile3 = net_cf[0,3+3+225+3+225:3+225+3+225+3+225,:3]

        rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)
        rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)
        tac_pose1 = rigid_body_states[0,3:3+225,:3]
        tac_pose2 = rigid_body_states[0,3+3+225:3+225+3+225,:3]
        tac_pose3 = rigid_body_states[0,3+3+225+3+225:3+225+3+225+3+225,:3]

        actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        if step == sim_length:
            tactile_list.append(np.concatenate([np.array(tactile1.cpu()),np.array(tactile2.cpu()),np.array(tactile3.cpu())]))
            tac_pose_list.append(np.concatenate([np.array(tac_pose1.cpu()),np.array(tac_pose2.cpu()),np.array(tac_pose3.cpu())]))
            object_pos_list.append(np.array(rigid_body_states[0,-1,:7].cpu()))

            step = 0
            i=i+1
            print("time cost:", time.time() - t)
            t = time.time()
            print('episode:',i)
            # plot_tactile_heatmap(tactile1,tac_pose1)
            # plot_tactile_heatmap(tactile2,tac_pose2)
            # plot_tactile_heatmap(tactile3,tac_pose3)
            ###reset

            dof_state[:,:] = 0
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

            root_state_tensor[-1,:] = random_pose(noise_scale)
            object_indices = torch.tensor([3],device=f'cuda:{index}',dtype=torch.int32)
            gym.set_actor_root_state_tensor_indexed(sim,gymtorch.unwrap_tensor(root_state_tensor),
                                                         gymtorch.unwrap_tensor(object_indices), len(object_indices))

        gym.sync_frame_time(sim)

    return tactile_list, tac_pose_list, object_pos_list

def random_pose(noise_scale):
    pose = torch.tensor([0.1,0.1,object_height, 0,0,0,1, 0,0,0,0,0,0])
    rand_quat = quat_from_angle_axis(torch.rand(1)*np.pi*2, torch.tensor([0, 0, 1], dtype=torch.float))

    pose[3:7] = rand_quat
    # pose[-3:] = noise_scale * torch.rand(3)
    return pose

def plot_tactile_heatmap(tactile,tactile_pose,object_name=None):

    tac = np.abs(np.array(tactile.to('cpu')))
    tac_pose= np.array(tactile_pose.to('cpu'))

    u = tac[:,0].reshape(15,15)
    v = tac[:,1].reshape(15,15)
    w = tac[:,2].reshape(15,15)
    if np.sum(w)>0:

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        im = ax.imshow(w, cmap=plt.cm.hot_r)
        ax.set_xticks(range(15))
        ax.set_yticks(range(15))

        fig.colorbar(im)

        plt.show()

if __name__ == "__main__":

    tactile_list, tac_pose_list, object_pos_list=run_sim(if_viewer,sim_length, num_iter,noise_scale)

    save_dir = '../tac_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = {'tactile': np.array(tactile_list),
            'tac_pose': np.array(tac_pose_list),
            'object_pos': np.array(object_pos_list)
    }

    with open(save_dir+object_name+'_multi.pkl','wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

    # with open(save_dir+object_name+'.pkl','rb') as f:
    #     d = pickle.load(f)