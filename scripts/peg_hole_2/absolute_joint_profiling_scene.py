# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

import torch
import numpy as np
import os
ROBOT_ASSET_DIR = os.path.join(os.path.dirname(__file__), "usd")

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG  # isort:skip


@configclass
class JointProfilingSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    default_joint_pos_gripper = (np.array([1, 1.0, 1.0, 1, -0.8, -0.8], dtype=np.float32)*0.07).tolist()

    robot : ArticulationCfg = ArticulationCfg(
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

    # add lights
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))



class run_simulator:
    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Runs the simulation loop."""
        # Extract scene entities
        # note: we only do this here for readability.
        self.robot = scene["robot"]
        self.scene = scene
        self.sim = sim

        # Define simulation stepping
        self.sim_dt = sim.get_physics_dt()
        self.sim_step_counter = 0
        # Simulation loop

    def step(self):
        self.sim_step_counter += 1
        self.scene.write_data_to_sim()
        self.sim.step(render=False)

        self.scene.update(dt = self.sim_dt)


    def run(self):
        while simulation_app.is_running():
            # Reset
            if count % 500 == 0:
                # reset counter
                count = 0
                # reset the scene entities
                # root state
                # we offset the root state by the origin since the states are written in simulation world frame
                # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # set joint positions with some noise
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                joint_pos += torch.rand_like(joint_pos) * 0.1
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                scene.reset()
                print("[INFO]: Resetting robot state...")
            # Apply random action
            # -- generate random joint efforts
            efforts = torch.randn_like(robot.data.joint_pos) * 5.0
            # -- apply action to the robot
            robot.set_joint_effort_target(efforts)

            # Step simulation
            self.step()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
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

    sim = SimulationContext(sim_cfg)
    # Set main camera
    # sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = JointProfilingSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
