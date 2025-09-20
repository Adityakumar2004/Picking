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
### ------------
# import omni.ui as ui

### ------------


"""Rest everything follows."""
from typing import Union
import torch
import time
import numpy as np
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
# from utils_1 import env_wrapper, log_values
import os
from keyboard_interface import keyboard_custom
dt = 0.1


def make_env(video_folder:str | None =None, output_type: str = "numpy"):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="scripts.custom_scripts.robot_env:RobotEnv",
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


def main():
    env = make_env(video_folder=None, output_type="numpy")
    env.reset()

    keyboard = keyboard_custom(pos_sensitivity=1.0*args_cli.sensitivity, rot_sensitivity=1.0*args_cli.sensitivity)
    keyboard.reset()
    print(f"\n\n{keyboard}\n\n")

    num_envs = env.unwrapped.scene.num_envs
    while simulation_app.is_running():

        keyboard_output = keyboard.advance()

        pose_action = keyboard_output["pose_command"]
        close_gripper = keyboard_output["gripper_command"]
        recording_state = keyboard_output["recording_state"]

        if keyboard_output["reset_state"]:
            print("\n resetting the env \n ", "---"*10, "\n")
            env.reset()
        
        pose_action[:3] = pose_action[:3] * 0.03 
        pose_action[3:6] = pose_action[3:6] * 0.06
        
        if close_gripper:
            action = np.concatenate((pose_action, np.array([-1.0])), axis=0)
        else:
            action = np.concatenate((pose_action, np.array([1.0])), axis=0)

        actions = torch.from_numpy(action).float().repeat(env.unwrapped.scene.num_envs, 1)
        env.step(actions)
        print("action \n ", actions[0])

    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()
