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
import omni.ui as ui
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

from scripts.custom_scripts.robot_env import RobotEnv, RobotEnvCfg

# class ControllerWindow:
#     """A UI window for controlling robot parameters with sliders."""
    
#     def __init__(self, env:gym.Env=None):
#         """
#         Initialize the controller window.
        
#         Args:
#             initial_kp (float): Initial Kp value
#             initial_kd (float): Initial Kd value
#         """

#         self.slider_params = {}
#         self.env = env
#         # self.env = env
    
#     # Button to set the filename
#     def on_set_filename(self):
#         filename = self.filename_field.model.as_string
#         print(f"Logging to file: {filename}")
#         # Your logging code here
#         # start_logging(filename)

    
    
#     def create_window(self):
#         """Create the UI window and its contents."""
#         self.window = ui.Window("Controller Panel", width=300, height=200, 
#                                flags=ui.WINDOW_FLAGS_NO_COLLAPSE)
        
#         with self.window.frame:
#             with ui.VStack(spacing=10):
#                 for param_name in self.slider_params.keys():
#                     ui.Label(f"{param_name}")
#                     ui.FloatSlider(
#                         model=self.slider_params[param_name]['model'], 
#                         min=self.slider_params[param_name]['min'], 
#                         max=self.slider_params[param_name]['max']
#                     )
#                     ui.Spacer(height=10)
                
#         self.reset_window = ui.Window("reset_window", width=100, height=50, 
#                                flags=ui.WINDOW_FLAGS_NO_COLLAPSE)

#         with self.reset_window.frame:
#             with ui.VStack(spacing=10):
#                 ui.Label("Reset Parameters")
#                 ui.Button("Reset", clicked_fn=self._on_reset_clicked)

#                 ui.Label("log file name")
#                 self.filename_field = ui.StringField(
#                     model=ui.SimpleStringModel("test_log"),
#                 )

#                 ui.Button("Set Filename", clicked_fn=self.on_set_filename)





#         # Ensure window is visible
#         self.window.visible = True
#         print("UI Controller Panel created and should be visible")
#         # print(f"Initial values: kp={self.kp_model.as_float}, kd={self.kd_model.as_float}")

#     def _on_reset_clicked(self):
#         print("env reset ")
#         # self.env.reset()
#         env_ids = torch.arange(self.env.unwrapped.scene.num_envs, device=self.env.unwrapped.device)
#         joint_pos = self.env.unwrapped._robot.data.default_joint_pos[env_ids].clone()
#         joint_vel = torch.zeros_like(joint_pos)
#         self.env.unwrapped._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
#         self.env.unwrapped._robot.set_joint_position_target(joint_pos, env_ids = env_ids)
#         self.env.unwrapped._robot.set_joint_velocity_target(joint_vel, env_ids = env_ids)

#         # self.env.unwrapped.step_sim_no_action()
        

#     def _setup_callbacks(self):
#         """Setup callbacks for model changes."""
#         # self.models['kp'].add_value_changed_fn(self._on_kp_changed)
#         # self.models['kd'].add_value_changed_fn(self._on_kd_changed)
#         pass;

#     def get_params(self, param_name):
#         """Get current parameters as dictionary."""
#         return self.slider_params[param_name]["value"]
    
#     def set_visible(self, visible):
#         """Show or hide the window."""
#         if self.window:
#             self.window.visible = visible
    
#     def destroy(self):
#         """Clean up the window."""
#         if self.window:
#             self.window.destroy()
#             self.window = None

#     def create_new_slider_widget(self, param_name, value, min_val=0, max_val=1000, callback_fn=None, **kwargs):

#         if param_name in self.slider_params.keys():
#             print(f"Model {param_name} already exists.")
#             return
#         self.slider_params[param_name] = {
#             'model': ui.SimpleFloatModel(value),
#             'value': value,
#             'min': min_val,
#             'max': max_val,
#         }
#         callback_fn_model = lambda m: callback_fn(self.slider_params[param_name], m, **kwargs)
#         self.slider_params[param_name]['model'].add_value_changed_fn(callback_fn_model)


## callback_fn 
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

def callback_kd(params, model, env:gym.Env, group_name):
    new_kd_value = model.as_float
    joint_indices = env.unwrapped._robot.actuators[group_name].joint_indices
    # env.unwrapped._robot.actuators[group_name].damping[:,:] = new_kd_value
    env.unwrapped._robot.data.joint_damping[:,joint_indices] = new_kd_value
    damping_tensor = env.unwrapped._robot.data.joint_damping.clone()
    # env.unwrapped._robot.write_joint_stiffness_to_sim(damping_tensor, joint_indices)

    env_ids = torch.arange(env.unwrapped.scene.num_envs, device="cpu")
    env.unwrapped._robot.root_physx_view.set_dof_dampings(damping_tensor.cpu(), env_ids)
    params["value"] = model.as_float




def make_env(video_folder:str | None =None, output_type: str = "numpy"):

    id_name = "peg_insert-v0-uw"
    # gym.register(
    #     id=id_name,
    #     entry_point="scripts.peg_hole_2.factory_env_kinova:FactoryEnv",
    #     disable_env_checker=True,
    #     kwargs={
    #         "env_cfg_entry_point":"scripts.peg_hole_2.factory_env_cfg_diff_ik:FactoryTaskPegInsertCfg",
    #     },
    # )

    # gym.register(
    #     id=id_name,
    #     entry_point="scripts.peg_hole_2.robot_env:RobotEnvDiffIK",
    #     disable_env_checker=True,
    #     kwargs={
    #         "env_cfg_entry_point":"scripts.peg_hole_2.robot_env:RobotEnvCfgDiffIK",
    #     },
    # )

    gym.register(
        id=id_name,
        entry_point="scripts.peg_hole_2.factory_env_diff_ik:FactoryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"scripts.peg_hole_2.factory_env_cfg_diff_ik:FactoryTaskPegInsertCfg",
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


    # controller_window = ControllerWindow(env)
    
    # controller_window.create_new_slider_widget(
    #     param_name="kp_arm1",
    #     value = 800.0,
    #     min_val=0,
    #     max_val=1000000,
    #     callback_fn=callback_kp,
    #     env=env,
    #     group_name="kinova_shoulder"
    # )
    # controller_window.create_new_slider_widget(
    #     param_name="kd_arm1",
    #     value = 160.0,
    #     min_val=0,
    #     max_val=1000000,
    #     callback_fn=callback_kd,
    #     env=env,
    #     group_name="kinova_shoulder"
    # )

    # controller_window.create_new_slider_widget(
    #     param_name="kp_arm2",
    #     value = 800.0,
    #     min_val=0,
    #     max_val=1000000,
    #     callback_fn=callback_kp,
    #     env=env,
    #     group_name="kinova_forearm"
    # )
    # controller_window.create_new_slider_widget(
    #     param_name="kd_arm2",
    #     value = 160.0,  
    #     min_val=0,
    #     max_val=1000000,
    #     callback_fn=callback_kd,
    #     env=env,
    #     group_name="kinova_forearm"
    # )

    # controller_window.create_window()
    
    num_envs = env.unwrapped.scene.num_envs

    while simulation_app.is_running():

        keyboard_output = keyboard.advance()

        pose_action = keyboard_output["pose_command"]
        close_gripper = keyboard_output["gripper_command"]
        recording_state = keyboard_output["recording_state"]

        if keyboard_output["reset_state"]:
            print("\n resetting the env \n ", "---"*10, "\n")
            env.reset()
        
        # pose_action[:3] = pose_action[:3] * 0.03 
        # pose_action[3:6] = pose_action[3:6] * 0.06
        

        # if close_gripper:
        #     action = np.concatenate((pose_action, np.array([-1.0])), axis=0)
        # else:
        #     action = np.concatenate((pose_action, np.array([1.0])), axis=0)

        # actions = torch.from_numpy(action).float().repeat(env.unwrapped.scene.num_envs, 1)
        actions = torch.from_numpy(pose_action).float().repeat(env.unwrapped.scene.num_envs, 1)

        env.step(actions)
        print("action \n ", actions[0])

    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()
