import argparse

import os
import numpy as np

# Initialize Isaac Sim before other imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=" keyboard teleop")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules after app is launched
import gymnasium as gym
import json
import pandas as pd
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import omni.ui as ui

import torch
import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

ROBOT_ASSET_DIR = os.path.join(os.path.dirname(__file__), "usd")

@configclass
class JointProfilingEnvCfg(DirectRLEnvCfg):
    default_joint_pos_gripper = (np.array([1, 1.0, 1.0, 1, -0.8, -0.8], dtype=np.float32)*0.07).tolist()
    num_envs: int = args_cli.num_envs
    episode_length_s: int = 20*60*60
    decimation: int = 1
    action_space = 7 ## dof
    observation_space = 3 ## randomly set
    state_space = 3 ## randomly set

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOT_ASSET_DIR, "Robots/Kinova/gen3n7.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "gen3_joint_1": -0.0390,
                "gen3_joint_2": 0.8417,
                "gen3_joint_3": -0.0531,
                "gen3_joint_4": 2.2894,
                "gen3_joint_5": -0.0744,
                "gen3_joint_6": -1.5667,
                "gen3_joint_7": -1.5310,
                "finger_joint": default_joint_pos_gripper[0], #0.4, # 0, 0.8
                "left_inner_knuckle_joint": default_joint_pos_gripper[1], #0.1, # 0, 0.8757
                "right_inner_knuckle_joint": default_joint_pos_gripper[2], #0.1, # 0, 0.8757
                "right_outer_knuckle_joint": default_joint_pos_gripper[3], #0.1, # 0, 0.81
                "left_inner_finger_joint": default_joint_pos_gripper[4], #-0.08, #-0.8757, 0
                "right_inner_finger_joint": default_joint_pos_gripper[5], #-0.08, #-0.8757, 0
            },
        ),
        actuators={
            "kinova_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["gen3_joint_[1-4]"],
                # effort_limit= 2,
                # velocity_limit=1.0,
                stiffness=800.0,
                damping=160.0,
                effort_limit=4000.0,
                velocity_limit=675,
                # stiffness=1800.0,
                # damping=10.0,
            ),
            "kinova_forearm": ImplicitActuatorCfg(
                joint_names_expr=["gen3_joint_[5-7]"], 
                # effort_limit= 2,
                # velocity_limit=1.0,#2.5,
                stiffness=800.0, 
                damping=160.0,
                effort_limit=4000.0,
                velocity_limit=675,
                # stiffness=1800.0,
                # damping=10.0,

            ),
            "kinova_gripper": ImplicitActuatorCfg(
                joint_names_expr=['finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint'],
                effort_limit=200.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=100.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    franka_fingerpad_length: float = 0.017608
    robot_friction :float = 0.75
 
class JointProfilingEnv(DirectRLEnv):
    cfg: JointProfilingEnvCfg
    def __init__(self, cfg: JointProfilingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._set_body_inertias()
        self._set_friction(self._robot, self.cfg.robot_friction)

        arm_joint_names = ["gen3_joint_.*"]
        self.arm_joint_ids = self._robot.find_joints(arm_joint_names)[0]

        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self._robot = Articulation(self.cfg.robot)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self._robot
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions):
        self.actions = actions
        self.ctrl_target_joint_pos = self._robot.data.joint_pos.clone()
        self._compute_intermediate_values(self.physics_dt)
    
    def _apply_action(self):
        desired_joint_pos = self.actions
        self.ctrl_target_joint_pos[:, self.arm_joint_ids] = desired_joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

    def _compute_intermediate_values(self, dt: float):

        self.current_joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

    def _get_dones(self):
        
        truncated = self.episode_length_buf >= self.max_episode_length-1
        return truncated, truncated
    
    def _get_observations(self):
        observations = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
        return observations
    

        ## setup_scene
        ## pre_physics_step() 
        ## action
        ## get_dones, rewards, observations, infos, reset_idx, 
        ## default_pose

    def _get_rewards(self):
        rewards = torch.zeros(self.num_envs, device=self.device)
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        # print('default joint pos \n', joint_pos.cpu().numpy())
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._compute_intermediate_values(self.physics_dt)

        self._compute_intermediate_values(self.physics_dt)

        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # print('joint pos after writing \n', self._robot.data.joint_pos.clone().cpu().numpy())

def make_env():

    id_name = "peg_insert-v0-uw"

    gym.register(
        id=id_name,
        entry_point=f"{__name__}:JointProfilingEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:JointProfilingEnvCfg",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array")
    return env

def main():
    env = make_env()
    env.reset()

    while True:
        # actions = torch.zeros((env.unwrapped.num_envs, 7), device=env.unwrapped.device)
        actions = env.unwrapped.current_joint_pos[:, env.unwrapped.arm_joint_ids].clone()
        env.step(actions)

if __name__ == "__main__":
    main()