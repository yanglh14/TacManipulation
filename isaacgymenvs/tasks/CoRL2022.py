from isaacgym import gymapi
from isaacgym import gymtorch


gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 50
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

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
asset_root = "../../assets"
# asset_file = "tactile/objects/ball.xml"
asset_file = "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf"

asset_options = gymapi.AssetOptions()

asset2 = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)


# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    gym.begin_aggregate(env, 300, 1000, True)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.1, 0.1, 0.1)
    actor_handle = gym.create_actor(env, asset1, pose, "Desk", i, 0)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.1, 0.1, 0.2)
    actor_handle = gym.create_actor(env, asset2, pose, "Object", i, 0)

    gym.end_aggregate(env)

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
cam_pos = gymapi.Vec3(0.3, 0.3, 0.3)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    _net_cf = gym.acquire_net_contact_force_tensor(sim)
    net_cf = gymtorch.wrap_tensor(_net_cf)

    rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)
    rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)
    tactile_pose = rigid_body_states[0,3:3+225,:3]

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
