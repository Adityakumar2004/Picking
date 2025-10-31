# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import os

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert
import numpy as np

## changes maded
# action bounds in CtrlCfg
# action thresholds in CtrlCfg 
# decimation in FactoryEnvCfg 8 --> 1
# episode length 

ROBOT_ASSET_DIR = os.path.join(os.path.dirname(__file__), "usd")


OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [5.5, 5.5, 5.5] #[0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [1.0, 1.0, 1.0]#[0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0
    kd_null = 6.3246

@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    default_joint_pos = np.array([1, 1.0, 1.0, 1, -0.8, -0.8])*0.07
    ## rounding the numbers and converting to list to avoid precision issues while comparing with the config
    default_joint_pos = np.round(default_joint_pos, decimals=4).tolist()
    decimation = 1
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s = 60*60*2 #10.0  # Probably need to override.
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
            # joint_pos={
            #     "gen3_joint_1": 0.00871,
            #     "gen3_joint_2": -0.10368,
            #     "gen3_joint_3": -0.00794,
            #     "gen3_joint_4": -1.49139,
            #     "gen3_joint_5": -0.00083,
            #     "gen3_joint_6": 1.38774,
            #     "gen3_joint_7": 0.0,
            #     "finger_joint": 0.04,
            #     "left_inner_knuckle_joint": 0.0, # 0, 0.8757
            #     "right_inner_knuckle_joint": 0.0, # 0, 0.8757
            #     "right_outer_knuckle_joint": 0.0, # 0, 0.81
            #     "left_inner_finger_joint": 0.0, #-0.8757, 0
            #     "right_inner_finger_joint": 0.0, #-0.8757, 0
            # },
            joint_pos={
                "gen3_joint_1": -0.0390,
                "gen3_joint_2": 0.8417,
                "gen3_joint_3": -0.0531,
                "gen3_joint_4": 2.2894,
                "gen3_joint_5": -0.0744,
                "gen3_joint_6": -1.5667,
                "gen3_joint_7": -1.5310,
                "finger_joint": default_joint_pos[0], #0.4, # 0, 0.8
                "left_inner_knuckle_joint": default_joint_pos[1], #0.1, # 0, 0.8757
                "right_inner_knuckle_joint": default_joint_pos[2], #0.1, # 0, 0.8757
                "right_outer_knuckle_joint": default_joint_pos[3], #0.1, # 0, 0.81
                "left_inner_finger_joint": default_joint_pos[4], #-0.08, #-0.8757, 0
                "right_inner_finger_joint": default_joint_pos[5], #-0.08, #-0.8757, 0
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

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
    

@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 20.0


@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0
