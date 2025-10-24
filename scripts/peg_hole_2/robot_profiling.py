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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab.sim as sim_utils

import os
from keyboard_interface import keyboard_custom
dt = 0.1

import sys

# Add the Isaac Lab root directory to Python path for module imports
ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ISAACLAB_ROOT not in sys.path:
    print(f"Adding Isaac Lab root to Python path: {ISAACLAB_ROOT}")
    sys.path.insert(0, ISAACLAB_ROOT)





class ControllerWindow:
    """A UI window for controlling robot parameters with sliders."""
    
    def __init__(self, env:gym.Env=None):
        """
        Initialize the controller window.
        
        Args:
            initial_kp (float): Initial Kp value
            initial_kd (float): Initial Kd value
        """

        self.slider_params = {}
        self.env = env
        # self.env = env
    
    def on_set_filename(self):
        # filename = self.filename_field.model.as_string
        filename = self.filename_model.as_string
        print(f"Logging to file: {filename}")
        # Your logging code here
        # start_logging(filename)

    def on_logging_clicked(self, model):
        logging_enabled = model.as_bool
        self.env.unwrapped.exp_name = self.filename_model.as_string
        if logging_enabled:
            print("Logging Enabled")
            self.env.unwrapped.recording_state = True
        else:
            print("Logging Disabled")
            self.env.unwrapped.recording_state = False
        print(f"logging enabled : {logging_enabled}")

    def on_trajectory_execute_clicked(self):
        global execute_trajectory
        execute_trajectory = not execute_trajectory
        if execute_trajectory:
            print("Executing Trajectory")
            # self._on_reset_clicked()
            # self.logging_model.as_bool = True
            # self.env.unwrapped.reset_logging(exp_name = self.filename_model.as_string, recording_state= True)
        else:
            print("Stopping Trajectory Execution")
            self.env.unwrapped.reset_logging(recording_state= False)
            self.env.unwrapped.save_to_csv()
            self.logging_model.as_bool = False
        
    def create_window(self):
        """Create the UI window and its contents."""
        self.window = ui.Window("Controller Panel", width=300, height=550, 
                               flags=ui.WINDOW_FLAGS_NO_COLLAPSE)        
        with self.window.frame:
            with ui.VStack():
                for param_name in self.slider_params.keys():
                    ui.Label(f"{param_name}")
                    ui.FloatSlider(
                        model=self.slider_params[param_name]['model'], 
                        min=self.slider_params[param_name]['min'], 
                        max=self.slider_params[param_name]['max']
                    )
                    ui.Spacer(height=3)
                    ui.FloatField(model=self.slider_params[param_name]['model'])
                    ui.Spacer(height=15)
                

        self.reset_window = ui.Window("reset_window", width=150, height=350, 
                               flags=ui.WINDOW_FLAGS_NO_COLLAPSE)
        with self.reset_window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Reset Parameters")
                ui.Button("Reset", clicked_fn=self._on_reset_clicked)

                ui.Label("log file name")
                # self.filename_field = ui.StringField(model=ui.SimpleStringModel("test_log"))
                ## will rather change the model value as that would change all the gui

                self.filename_model = ui.SimpleStringModel("test_log") 
                ui.StringField(model=self.filename_model)
                ui.Button("Set Filename", clicked_fn=self.on_set_filename)

                ui.Label("Enable Logging")
                self.logging_model = ui.SimpleBoolModel(False)
                self.logging_model.add_value_changed_fn(self.on_logging_clicked)
                ui.CheckBox(model=self.logging_model)


                ui.Button("Execute Trajectory", clicked_fn=self.on_trajectory_execute_clicked)


        # Ensure window is visible
        self.window.visible = True
        print("UI Controller Panel created and should be visible")
        # print(f"Initial values: kp={self.kp_model.as_float}, kd={self.kd_model.as_float}")

    def _on_reset_clicked(self):
        print("env reset ")
        global reset_env, execute_trajectory
        reset_env = True
        execute_trajectory = False
        self.logging_model.as_bool = False

        # self.env.reset()
        # env_ids = torch.arange(self.env.unwrapped.scene.num_envs, device=self.env.unwrapped.device)
        # joint_pos = self.env.unwrapped._robot.data.default_joint_pos[env_ids].clone()
        # joint_vel = torch.zeros_like(joint_pos)
        # joint_pos[:, :7] = torch.tensor(self.env.unwrapped.reset_joints_real_hardware, device=self.env.unwrapped.device)[None, :]
        # self.env.unwrapped._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # self.env.unwrapped._robot.set_joint_position_target(joint_pos, env_ids = env_ids)
        # self.env.unwrapped._robot.set_joint_velocity_target(joint_vel, env_ids = env_ids)
        # self.env.unwrapped._robot.reset()
        # self.env.unwrapped.rl_step_count = 0
        # self.env.unwrapped.log_data = []
        # for _ in range(3):
        #     self.env.unwrapped.step_sim_no_action()

    def _setup_callbacks(self):
        """Setup callbacks for model changes."""
        # self.models['kp'].add_value_changed_fn(self._on_kp_changed)
        # self.models['kd'].add_value_changed_fn(self._on_kd_changed)
        pass;

    def get_params(self, param_name):
        """Get current parameters as dictionary."""
        return self.slider_params[param_name]["value"]
    
    def set_visible(self, visible):
        """Show or hide the window."""
        if self.window:
            self.window.visible = visible
    
    def destroy(self):
        """Clean up the window."""
        if self.window:
            self.window.destroy()
            self.window = None

    def create_new_slider_widget(self, param_name, value, min_val=0, max_val=1000, callback_fn=None, **kwargs):

        if param_name in self.slider_params.keys():
            print(f"Model {param_name} already exists.")
            return
        self.slider_params[param_name] = {
            'model': ui.SimpleFloatModel(value),
            'value': value,
            'min': min_val,
            'max': max_val,
        }
        callback_fn_model = lambda m: callback_fn(self.slider_params[param_name], m, **kwargs)
        self.slider_params[param_name]['model'].add_value_changed_fn(callback_fn_model)


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

def clb_torque_limit(slider_params_dict, model, env:gym.Env, group_name:list[str]):
    new_torque_limit = model.as_float
    joint_indices = []
    for g_name in group_name:
        joint_indices.extend(env.unwrapped._robot.actuators[g_name].joint_indices)
    joint_indices = torch.tensor(joint_indices, device=env.unwrapped.device)
    env.unwrapped._robot.write_joint_effort_limit_to_sim(new_torque_limit, joint_indices)
    slider_params_dict["value"] = model.as_float

def clb_velocity_limit(slider_params_dict, model, env:gym.Env, group_name=list[str]):
    new_velocity_limit = model.as_float
    joint_indices = []
    for g_name in group_name:
        joint_indices.extend(env.unwrapped._robot.actuators[g_name].joint_indices)
    joint_indices = torch.tensor(joint_indices, device=env.unwrapped.device)
    env.unwrapped._robot.write_joint_velocity_limit_to_sim(new_velocity_limit, joint_indices)
    # slider_params_dict["value"] = model.as_float
    
def make_env(video_folder:str | None =None, output_type: str = "numpy"):

    id_name = "peg_insert-v0-uw"

    gym.register(
        id=id_name,
        entry_point="scripts.peg_hole_2.absolute_target_env:RobotEnvLogging",
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

    return env

def main():
    env = make_env(video_folder=None, output_type="numpy")
    env.unwrapped.enable_env_tune_changes(True)

    env.reset()

    keyboard = keyboard_custom(pos_sensitivity=1.0*args_cli.sensitivity, rot_sensitivity=1.0*args_cli.sensitivity)
    keyboard.reset()
    print(f"\n\n{keyboard}\n\n")


    controller_window = ControllerWindow(env)
    
    controller_window.create_new_slider_widget(
        param_name="kp_arm1",
        value = env.unwrapped._robot.data.joint_stiffness[:,env.unwrapped._robot.actuators["kinova_shoulder"].joint_indices].mean().item(),
        min_val=0,
        max_val=1000000,
        callback_fn=callback_kp,
        env=env,
        group_name="kinova_shoulder"
    )
    controller_window.create_new_slider_widget(
        param_name="kd_arm1",
        value = env.unwrapped._robot.data.joint_damping[:,env.unwrapped._robot.actuators["kinova_shoulder"].joint_indices].mean().item(),
        min_val=0,
        max_val=1000000,
        callback_fn=callback_kd,
        env=env,
        group_name="kinova_shoulder"
    )
    controller_window.create_new_slider_widget(
        param_name="kp_arm2",
        value = env.unwrapped._robot.data.joint_stiffness[:,env.unwrapped._robot.actuators["kinova_forearm"].joint_indices].mean().item(),
        min_val=0,
        max_val=1000000,
        callback_fn=callback_kp,
        env=env,
        group_name="kinova_forearm"
    )
    controller_window.create_new_slider_widget(
        param_name="kd_arm2",
        value = env.unwrapped._robot.data.joint_damping[:,env.unwrapped._robot.actuators["kinova_forearm"].joint_indices].mean().item(),
        min_val=0,
        max_val=1000000,
        callback_fn=callback_kd,
        env=env,
        group_name="kinova_forearm"
    )
    controller_window.create_new_slider_widget(
        param_name="torque_limit",
        # value = env.unwrapped._robot.actuators["kinova_shoulder"].effort_limit[0,0].item(),
        value = env.unwrapped._robot.data.joint_effort_limits[0,0].item(),
        min_val=0,
        max_val=5000,
        callback_fn=clb_torque_limit,
        env=env,
        group_name=["kinova_shoulder", "kinova_forearm"]
    )
    controller_window.create_new_slider_widget(
        param_name="velocity_limit",
        value= env.unwrapped._robot.data.joint_velocity_limits[0,0].item(),
        min_val=0,
        max_val=800,
        env=env,
        callback_fn=clb_velocity_limit,
        group_name=["kinova_shoulder", "kinova_forearm"]
    )

    controller_window.create_window()
    
    num_envs = env.unwrapped.scene.num_envs
    global execute_trajectory, reset_env
    reset_env = False
    execute_trajectory = False

    step = 0
    recording_step = 0
    while simulation_app.is_running():

        keyboard_output = keyboard.advance()

        pose_action = keyboard_output["pose_command"]
        close_gripper = keyboard_output["gripper_command"]
        recording_state = keyboard_output["recording_state"]

        if keyboard_output["reset_state"]:
            print("\n resetting the env \n ", "---"*10, "\n")
            env.reset()
        
        pose_action[:3] = pose_action[:3] * 0.1 
        pose_action[3:6] = pose_action[3:6] * 0.097

        if execute_trajectory:
            if recording_step == 0:
                env.reset()
                # for _ in range(40):
                #     actions = torch.zeros((env.unwrapped.scene.num_envs, 6), dtype=torch.float32, device=env.unwrapped.device)
                #     env.unwrapped.step(actions)
                env.unwrapped.reset_logging(exp_name = controller_window.filename_model.as_string, recording_state= True)
                controller_window.logging_model.as_bool = True

    
            pose_action = np.zeros(6, dtype=np.float32)
            pose_action[0] = 0.1
            recording_step += 1

            if recording_step >= 180:
                recording_step = 0
                execute_trajectory = False
                env.unwrapped.save_to_csv()
                env.unwrapped.reset_logging(recording_state= False)
                controller_window.logging_model.as_bool = False
                print("Stopping Trajectory Execution after completing the trajectory")
        
        else:
            recording_step = 0

        if reset_env:
            env.reset()
            reset_env = False

        actions = torch.from_numpy(pose_action).float().repeat(env.unwrapped.scene.num_envs, 1)
        next_obs, reward, terminated, truncated, info_custom = env.step(actions)

        print("----"*8, "simulation step ",step, "----"*8)
        # print("velocity limit \n", env.unwrapped._robot.data.joint_vel_limits)
        print("torque limit \n", env.unwrapped._robot.data.joint_effort_limits)
        print("arm joint ids \n", env.unwrapped.arm_joint_ids)
        joint_indices = []
        for g_name in ["kinova_shoulder", "kinova_forearm"]:
            joint_indices.extend(env.unwrapped._robot.actuators[g_name].joint_indices)
        print("group ids \n", joint_indices)
        print("applied torques \n", env.unwrapped._robot.data.applied_torque[0, :7])
        step+=1

    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()
