import os.path
import pickle
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

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

gym.prepare_sim(sim)
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
asset_file = "tactile/CoRL2022/corl2022.xml"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01

asset1 = gym.load_asset(sim, asset_root, asset_file, asset_options)

#### load object asset
if_viewer = False
sim_length = 100
num_iter = 1000
noise_scale = 0.05
object_name = 'cylinder_small'
asset_root = "../../assets"
# asset_file = "tactile/objects/ycb/"+object_name+"/"+object_name+ ".urdf"
asset_file = "tactile/objects/"+ object_name +".urdf"

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01

asset2 = gym.load_asset(sim, asset_root, asset_file, asset_options)

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
        gym.begin_aggregate(env, 300, 1000, True)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.1, 0.1, 0.1)
        actor_handle0 = gym.create_actor(env, asset1, pose, "Desk", i, 0)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.1, 0.1, 0.12)
        actor_handle1 = gym.create_actor(env, asset2, pose, "Object", i, 0)

        gym.end_aggregate(env)

    if if_viewer:
        cam_props = gymapi.CameraProperties()
        viewer = gym.create_viewer(sim, cam_props)
        cam_pos = gymapi.Vec3(0.3, 0.3, 0.3)
        cam_target = gymapi.Vec3(0, 0, 0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    f,p = [],[]
    step, i = 0,0
    tactile_list, tac_pose_list, object_pos_list = [],[],[]

    # while not gym.query_viewer_has_closed(viewer):
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

        # get pose and tactile
        _net_cf = gym.acquire_net_contact_force_tensor(sim)
        net_cf = gymtorch.wrap_tensor(_net_cf).view(1, -1, 3)
        tactile = net_cf[0,3:3+225,:3]

        rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)
        rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)
        tac_pose = rigid_body_states[0,3:3+225,:3]

        actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        if step == sim_length:
            tactile_list.append(np.array(tactile))
            tac_pose_list.append(np.array(tac_pose))
            object_pos_list.append(np.array(rigid_body_states[0,-1,:7]))

            step = 0
            i=i+1
            ###reset object
            root_state_tensor[-1,:] = random_pose(noise_scale)
            object_indices = torch.tensor([root_state_tensor.shape[0]-1],dtype=torch.int32)
            gym.set_actor_root_state_tensor_indexed(sim,gymtorch.unwrap_tensor(root_state_tensor),
                                                         gymtorch.unwrap_tensor(object_indices), len(object_indices))

        #     plot_tactile_heatmap(tactile,tac_pose,object_name)
        #     plot_tactile_heatmap(tactile,tac_pose)

        #### get contacts information
        # contacts =gym.get_env_rigid_contacts(env)
        # a = []
        # for i in range(contacts.shape[0]):
        #     if contacts[i]['body1'] == 115:
        #         a.append(contacts[i]['body1'])
        #         p.append(contacts[i]['initialOverlap'])
        #         f.append( np.linalg.norm(np.array(tactile[112,:3])) )

        gym.sync_frame_time(sim)

    # plot_forceposition(p, f)
    return tactile_list, tac_pose_list, object_pos_list

def random_pose(noise_scale):
    pose = torch.tensor([0.1,0.1,0.15, 0,0,0,1, 0,0,0,0,0,0])
    rand_quat = quat_from_angle_axis(torch.rand(1)*np.pi, torch.tensor([0, 0, 1], dtype=torch.float))
    pose[3:7] = rand_quat
    pose[:3] = pose[:3] + noise_scale * torch.rand(3)
    return pose

def plot_forceposition(p,f):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(0.4+np.array(p)*100,label='position')
    ax.plot(np.array(f)*10,label='force')

    ax.legend()
    plt.show()

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
        if object_name != None:
            plt.savefig(os.path.abspath(os.path.join(os.getcwd(),'../..')) + "/Pictures/"+object_name+".png")
            plt.close(fig)
        plt.show()

def plot_tactile_streamplot(tactile,tactile_pose):

    tac = np.abs(np.array(tactile.to('cpu')))
    tac_pose= np.array(tactile_pose.to('cpu'))

    y, x = np.mgrid[0:0.15:15j, 0:0.15:15j]

    u = tac[:,0].reshape(15,15)
    v = tac[:,1].reshape(15,15)
    w = tac[:,2].reshape(15,15)

    if np.sum(u+v)>0:

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        strm = ax.streamplot(x, y, u, v, density=1, color=u+v, linewidth=2, cmap='autumn')
        fig.colorbar(strm.lines)

        ax.quiver(x, y, u, v)

        plt.show()

def plot_tactile_3d(tactile,tactile_pose):

    tac = np.abs(np.array(tactile.to('cpu')))
    tac_pose= np.array(tactile_pose.to('cpu'))

    x = tac_pose[:, 0]
    y = tac_pose[:, 1]
    z = tac_pose[:, 2]
    u = tac[:,0]
    v = tac[:,1]
    w = tac[:,2]
    colors = np.concatenate((  np.clip((w.reshape([225,1])*10),0,1)
                             ,np.zeros([225,1]),np.zeros([225,1]),np.ones([225,1])),axis=1)

    if np.sum(u+v+w) >0:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.quiver(x, y, z, u, v, w,length=0.5,arrow_length_ratio=0.2,linewidths =2.5)
        # ax.quiverkey(q,X=0.3,Y=1.1,U=10,label='Quiver key', labelpos='E')
        ax.scatter(x, y, z,s=10)

        ax.set_xlim(0, 0.2)
        ax.set_ylim(0, 0.2)
        ax.set_zlim(0.1, 0.3)

        plt.show()
        print('step+1')

if __name__ == "__main__":
    tactile_list, tac_pose_list, object_pos_list=run_sim(if_viewer,sim_length, num_iter,noise_scale)

    save_dir = '../tac_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = {'tactile': np.array(tactile_list),
            'tac_pose': np.array(tac_pose_list),
            'object_pos': np.array(object_pos_list)
    }

    with open(save_dir+object_name+'.pkl','wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

    # with open(save_dir+object_name+'.pkl','rb') as f:
    #     d = pickle.load(f)