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

def make_env(video_folder:str | None =None, output_type: str = "numpy") -> RobotEnv:

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="scripts.custom_scripts.robot_env:RobotEnvLogging",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"scripts.custom_scripts.robot_env:RobotEnvCfg",
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

    if rl_step == 1:
        actions[env_ids, 0:3] = 0.03

    return actions



def main():
    env = make_env()
    env.reset()

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
