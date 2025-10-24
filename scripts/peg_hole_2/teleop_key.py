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
from utils_1 import env_wrapper, log_values
import os
from keyboard_interface import keyboard_custom
dt = 0.1

import sys

# Add the Isaac Lab root directory to Python path for module imports
ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ISAACLAB_ROOT not in sys.path:
    print(f"Adding Isaac Lab root to Python path: {ISAACLAB_ROOT}")
    sys.path.insert(0, ISAACLAB_ROOT)


# from scripts.peg_hole_2.robot_env import RobotEnv, RobotEnvCfg

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
    
    # Button to set the filename
    def on_set_filename(self):
        filename = self.filename_field.model.as_string
        print(f"Logging to file: {filename}")
        # Your logging code here
        # start_logging(filename)

    
    
    def create_window(self):
        """Create the UI window and its contents."""
        self.window = ui.Window("Controller Panel", width=300, height=200, 
                               flags=ui.WINDOW_FLAGS_NO_COLLAPSE)
        
        with self.window.frame:
            with ui.VStack(spacing=10):
                for param_name in self.slider_params.keys():
                    ui.Label(f"{param_name}")
                    ui.FloatSlider(
                        model=self.slider_params[param_name]['model'], 
                        min=self.slider_params[param_name]['min'], 
                        max=self.slider_params[param_name]['max']
                    )
                    ui.Spacer(height=10)
                
        self.reset_window = ui.Window("reset_window", width=100, height=50, 
                               flags=ui.WINDOW_FLAGS_NO_COLLAPSE)

        with self.reset_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Reset Parameters")
                ui.Button("Reset", clicked_fn=self._on_reset_clicked)

                ui.Label("log file name")
                self.filename_field = ui.StringField(
                    model=ui.SimpleStringModel("test_log"),
                )

                ui.Button("Set Filename", clicked_fn=self.on_set_filename)





        # Ensure window is visible
        self.window.visible = True
        print("UI Controller Panel created and should be visible")
        # print(f"Initial values: kp={self.kp_model.as_float}, kd={self.kd_model.as_float}")

    def _on_reset_clicked(self):
        print("env reset ")
        # self.env.reset()
        env_ids = torch.arange(self.env.unwrapped.scene.num_envs, device=self.env.unwrapped.device)
        joint_pos = self.env.unwrapped._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, :7] = torch.tensor(self.env.unwrapped.reset_joints_real_hardware, device=self.env.unwrapped.device)[None, :]
        self.env.unwrapped._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.env.unwrapped._robot.set_joint_position_target(joint_pos, env_ids = env_ids)
        self.env.unwrapped._robot.set_joint_velocity_target(joint_vel, env_ids = env_ids)
        self.env.unwrapped._robot.reset()

        # self.env.unwrapped.step_sim_no_action()
        

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
        entry_point="scripts.peg_hole_2.absolute_target_env:FactoryEnv",
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



def visualize_markers(env_marker_visualizer:VisualizationMarkers, pose: Union[np.ndarray, torch.Tensor], quat = None):
    '''
    Args:
    pose: tensor or numpy array of positions with dim --> (num_envs, num_spheres, 3) or (num_envs, 3) 
    
    '''
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    elif isinstance(pose, np.ndarray):
        pass
    else :
        assert False, "pose must be a torch.Tensor or np.ndarray"
    
    if quat is None:
        identity_quat = np.array([1, 0, 0, 0])  # identity quat for sphere

    if len(pose.shape) == 3:
        (num_envs, num_spheres, _) = pose.shape
    elif len(pose.shape) == 2:
        (num_envs, _) = pose.shape
        num_spheres = 1
        pose = pose[:, None, :]  # Add extra dimension to make it (num_envs, 1, _)
    else:
        raise ValueError(f"pose must have 2 or 3 dimensions, got shape {pose.shape}")

    if quat is not None:    
        if len(quat.shape) == 3:
            (num_envs, num_spheres, _) = quat.shape
        elif len(quat.shape) == 2:
            (num_envs, _) = quat.shape
            num_spheres = 1
            quat = quat[:, None, :]  # Add extra dimension to make it (num_envs, 1, _)
        else:
            raise ValueError(f"pose must have 2 or 3 dimensions, got shape {quat.shape}")

    translations = np.empty((num_envs * num_spheres, 3), dtype=np.float32)
    orientations = np.empty((num_envs * num_spheres, 4), dtype=np.float32)
    marker_indices = np.empty((num_envs * num_spheres,), dtype=np.int32)

    for env_id in range(num_envs):
        for count in range(num_spheres):
            translations[(num_spheres*env_id + count)] = pose[env_id, count, :3]
            if quat is None:
                orientations[(num_spheres*env_id + count)] = identity_quat
            else:
                orientations[(num_spheres*env_id + count)] = quat[env_id, count, :]
            marker_indices[(num_spheres*env_id + count)] = count

    env_marker_visualizer.visualize(
        translations=translations,
        orientations=orientations,
        marker_indices=marker_indices
    )

def create_marker_frames(count=1, scale=(0.01,0.01,0.01)):
    frame_markers = {}
    for i in range(count):
        frame_markers[f"frame_{i}"] = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=scale,
            )
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FrameMarkers",
        markers = frame_markers
    )

    marker_visualizer = VisualizationMarkers(marker_cfg)

    return marker_visualizer


def main():
    env = make_env(video_folder=None, output_type="numpy")
    env.unwrapped.enable_env_tune_changes(True)
    env.reset()

    keyboard = keyboard_custom(pos_sensitivity=1.0*args_cli.sensitivity, rot_sensitivity=1.0*args_cli.sensitivity)
    keyboard.reset()
    print(f"\n\n{keyboard}\n\n")


    controller_window = ControllerWindow(env)
    
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

    controller_window.create_window()
    
    num_envs = env.unwrapped.scene.num_envs

    ## visualizing markers
    frame_fingtip_viz = create_marker_frames(count=1, scale=(0.04,0.04,0.04))
    frame_bracelet_viz = create_marker_frames(count=1, scale=(0.04,0.04,0.04))
    frame_eef_viz = create_marker_frames(count=1, scale=(0.04,0.04,0.04))

    step = 0
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
        

        # if close_gripper:
        #     action = np.concatenate((pose_action, np.array([-1.0])), axis=0)
        # else:
        #     action = np.concatenate((pose_action, np.array([1.0])), axis=0)

        # actions = torch.from_numpy(action).float().repeat(env.unwrapped.scene.num_envs, 1)
        actions = torch.from_numpy(pose_action).float().repeat(env.unwrapped.scene.num_envs, 1)

        next_obs, reward, terminated, truncated, info_custom = env.step(actions)
        fingertip_pos = env.unwrapped.fingertip_midpoint_pos.clone().cpu().numpy() + env.unwrapped.scene.env_origins.cpu().numpy()
        fingertip_quat = env.unwrapped.fingertip_midpoint_quat.clone().cpu().numpy()

        bracelet_pos = env.unwrapped.logging_dict["gen3_bracelet_link_pos"]
        bracelet_quat = env.unwrapped.logging_dict["gen3_bracelet_link_quat"]

        end_effector_pos = env.unwrapped.logging_dict["gen3_end_effector_link_pos"]
        end_effector_quat = env.unwrapped.logging_dict["gen3_end_effector_link_quat"]

        eef_tip_linvel = env.unwrapped.logging_dict["ee_linvel"]
        eef_tip_angvel = env.unwrapped.logging_dict["ee_angvel"]

        ## finding the distance between the bracelet and the fingertip
        dist_vector_ee_bracelet = fingertip_pos - bracelet_pos
        # print("distance vector from bracelet to fingertip wtr world (base): \n", dist_vector_ee_bracelet)

        ## finding the distance from end effector_link to the fingertip
        dist_vector_ee_fingertip = fingertip_pos - end_effector_pos
        # print("distance vector from end_effector_link to fingertip wtr world (base): \n", dist_vector_ee_fingertip)

        visualize_markers(frame_eef_viz, end_effector_pos, end_effector_quat)
        visualize_markers(frame_bracelet_viz, bracelet_pos, bracelet_quat)
        visualize_markers(frame_fingtip_viz, fingertip_pos, fingertip_quat)
        print("---"*10, step, "---"*10)
        print("action ", actions[0].numpy())
        print("eef linvel ", eef_tip_linvel[0])
        print("eef angvel ", eef_tip_angvel[0])
        # print("fingertip pos \n", fingertip_pos[0])
        # print("fingertip quat \n", fingertip_quat[0])
        # print("obs policy \n", next_obs['policy'])
        # print("action \n ", actions[0])
        step+=1

    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()
