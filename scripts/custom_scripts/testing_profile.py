import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=" keyboard teleop")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# parser.add_argument("--video", action="store_true", help="Enable video recording during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=400, help="Interval between video recordings (in steps).")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# add argparse arguments
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



import gymnasium as gym
import numpy as np
import torch
import os 
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab.sim as sim_utils
from scripts.custom_scripts.robot_env import RobotEnv, RobotEnvCfg, RobotEnvLogging

from isaaclab.sensors import CameraCfg, Camera
from isaaclab.envs.ui import ViewportCameraController

def make_env(log_file, video_folder:str | None =None, output_type: str = "numpy") -> RobotEnv:

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="scripts.custom_scripts.robot_env:RobotEnvLogging",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"scripts.custom_scripts.robot_env:RobotEnvCfg",
            "log_file":log_file
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array")
     
    # env = env_wrapper(env, video_folder, output_type=output_type, enable_normalization_rewards=False)
    
    return env

# class env_wrapper(gym.Wrapper):
#     def __init__(self, env:RobotEnv):
#         super().__init__(env)
#         self.env = env

#         self.reset_logs()

    
#     def step(self, )

#     def reset_logs(self):
#         self.log_dict = {}
#         self.log_dict["ee_pos"] = []
#         self.log_dict["ee_quat"] = []
#         self.log_dict["joint_pos"] = []

#     def log_data(self):
#         self.log_dict["ee_pos"].append(self.env.current_ef_pos.clone().cpu().numpy())
#         self.log_dict["ee_quat"].append(self.env.current_ef_quat.clone().cpu().numpy())
#         self.log_dict["joint_pos"].append(self.env.joint_pos.clone().cpu().numpy())
# class camera:
#     def __init__(self, env:RobotEnvLogging):
#         self.env = env
#         self.cameras = []
#         self.vid_writers = []

#     def create_camera(self, pos, rot_quat, name="camera1"):
#         camera_cfg = CameraCfg(
#             prim_path=f"/World/{name}",
#             update_period=0.1,
#             height=480,
#             width=640,
#             data_types=["rgb"],
#             spawn=sim_utils.PinholeCameraCfg(
#                 focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
#             ),
#             offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.01), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),

#         )
#         camera = Camera(camera_cfg)
#         self.cameras.append(camera)
#         return camera

#     def record_cameras(self):
#         frame_dict = {}
#         for i, camera in enumerate(self.cameras):

#             cam_image = camera.data.output["rgb"]
#             if isinstance(cam_image, torch.Tensor):
#                 cam_image = cam_image.detach().cpu().numpy()
            
#             ## removing batch dimension
#             if cam_image.shape[0] == 1:
#                 cam_image = cam_image[0]

#             frame_dict[f"camera_{i}"] = cam_image

#             cam_image = cam_image.astype(np.uint8)
#             self.vid_writers[i].append_data(cam_image)

#         return frame_dict

#     def set_camera_pose_fixed_asset(self, camera_id, env_id):

#         # fixed_asset_default_state =  self.env.unwrapped._fixed_asset.data.default_root_state[env_id].clone()
        
#         camera_target = torch.tensor([0.0, 0.0, 0.005], device=self.env.unwrapped.device) + self.env.scene.env_origins[env_id]
#         eye_camera = camera_target + torch.tensor([0.5, -0.9, 0.3], device= self.env.unwrapped.device)

#         self.cameras[camera_id].set_world_poses_from_view(eye_camera.unsqueeze(0), camera_target.unsqueeze(0))



def get_current_ee_pose(env:RobotEnvLogging, env_ids=None):
    if env_ids is None:
        env_ids = np.arange(env.unwrapped.scene.num_envs)
    
    ee_pos = env.current_ef_pos[env_ids]
    ee_quat = env.current_ef_quat[env_ids]
    return ee_pos, ee_quat


def generate_actions(env:RobotEnv, rl_step, env_ids = None):
    '''
    a trajectory generated with action commands which are delta poses
    
    '''


    
    if env_ids is None:
        env_ids = np.arange(env.unwrapped.scene.num_envs)

    ## making tha actions of other envs not listed in envids zero
    actions = np.zeros((env.unwrapped.scene.num_envs, 7), dtype=np.float32)




    if rl_step >= 0 and rl_step <= 10:
        actions[env_ids, 0] = 0.05

    return actions

def callback_kp(params, model, env:gym.Env, group_name:str):
    '''
    both approaches will work isaac lab made a wrapper for setting stiffness to the physx sim 
    the other approach is to directly set the stiffness to the physx sim
    '''

    new_kp_value = model.as_float
    joint_names = env.unwrapped._robot.actuators[group_name].joint_names
    joint_indices = env.unwrapped._robot.actuators[group_name].joint_indices
    # env.unwrapped._robot.actuators[group_name].stiffness[:,:] = new_kp_value
    env.unwrapped._robot.data.joint_stiffness[:,joint_indices] = new_kp_value
    stiffness_tensor = env.unwrapped._robot.data.joint_stiffness.clone()

    ## approach 1: using the wrapper function
    # env.unwrapped._robot.write_joint_stiffness_to_sim(stiffness_tensor, joint_indices)
    
    ## approach 2: directly setting to physx sim
    env_ids = torch.arange(env.unwrapped.scene.num_envs, device="cpu")
    env.unwrapped._robot.root_physx_view.set_dof_stiffnesses(stiffness_tensor.cpu(), env_ids)
    params["value"] = model.as_float



def set_kp_kd(env:RobotEnvLogging, kp=None, kd=None, group_name=None, env_ids=None):
    if group_name is None:
        group_name = ["kinova_shoulder", "kinova_forearm"]

    if env_ids is None:
        env_ids = torch.arange(env.unwrapped.scene.num_envs, device="cpu")

    
    for group in group_name:
        
        joint_indices = env.unwrapped._robot.actuators[group].joint_indices
        if kp is not None:
            env.unwrapped._robot.data.joint_stiffness[: ,joint_indices] = kp
        if kd is not None:
            env.unwrapped._robot.data.joint_damping[:, joint_indices] = kd


        stiffness_tensor = env.unwrapped._robot.data.joint_stiffness.clone()
        
        env.unwrapped._robot.root_physx_view.set_dof_stiffnesses(stiffness_tensor.cpu(), env_ids)
    

    


def main():
    exp_name = "kp_5000_kd_0"
    log_dir = "scripts/custom_scripts/logs/csv_files"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}.csv")

    env = make_env(log_file)
    env.reset()

    # set_kp_kd(env, kp=5000.0, kd=0.0, group_name=["kinova_shoulder", "kinova_forearm"])
    
    rl_step = 0
    while simulation_app.is_running():
        actions = generate_actions(env, rl_step)
        actions_tensor = torch.from_numpy(actions).float()
        env.step(actions_tensor)
        print(actions)
        rl_step+= 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
