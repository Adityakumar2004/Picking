from datetime import datetime
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg, AssetBase, RigidObjectCfg, RigidObject
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, spawn_from_usd
from isaaclab.sim.spawners.lights import spawn_light
import os
import torch



from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

from isaaclab.utils.math import matrix_from_quat, quat_inv
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import axis_angle_from_quat
import isaacsim.core.utils.torch as torch_utils
# import custom_scripts.testing_controllers.controller as controller
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import pandas as pd

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "usd")

@configclass
class RobotEnvCfgDiffIK(DirectRLEnvCfg):

    episode_length_s = 60*60*10 ## 20sec = (rl_steps)*decimation*step_dt ## naming shouldnt be changed
    decimation = 20#20  ## decimation 20 means after every 20 sim steps one rl steps
    action_space = 7 ## 6- pose 1- gripper
    observation_space = 3 ## randomly set
    state_space = 3 ## randomly set


    '''
    physical_material
    This acts as the fallback material for any rigid body in the scene that doesnt explicitly override its material.

    Its also convenient when you want all objects to share the same baseline friction (and then later apply randomization on top).
    '''
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
        render=sim_utils.RenderCfg(
            enable_shadows=True,
            enable_reflections=True,
            enable_direct_lighting=True,
            samples_per_pixel=4,
            enable_ambient_occlusion=True,
            antialiasing_mode="DLAA"
        )
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)
    
    robot_friction: float = 0.75

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "Robots/Kinova/gen3n7.usd"),
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
            #     "gen3_joint_1": -0.0390,
            #     "gen3_joint_2": 0.8417,
            #     "gen3_joint_3": -0.0531,
            #     "gen3_joint_4": 2.2894,
            #     "gen3_joint_5": -0.0744,
            #     "gen3_joint_6": -1.5667,
            #     "gen3_joint_7": -1.5310,
            #     "finger_joint": 0.04, # 0, 0.8
            #     "left_inner_knuckle_joint": 0.1, # 0, 0.8757
            #     "right_inner_knuckle_joint": 0.1, # 0, 0.8757
            #     "right_outer_knuckle_joint": 0.1, # 0, 0.81
            #     "left_inner_finger_joint": -0.08, #-0.8757, 0
            #     "right_inner_finger_joint": -0.08, #-0.8757, 0
            # },

            joint_pos={
                "gen3_joint_1": 0.00871,
                "gen3_joint_2": -0.10368,
                "gen3_joint_3": -0.00794,
                "gen3_joint_4": -1.49139,
                "gen3_joint_5": -0.00083,
                "gen3_joint_6": 1.38774,
                "gen3_joint_7": 0.0,
                "finger_joint": 0.04,
                "left_inner_knuckle_joint": 0.0, # 0, 0.8757
                "right_inner_knuckle_joint": 0.0, # 0, 0.8757
                "right_outer_knuckle_joint": 0.0, # 0, 0.81
                "left_inner_finger_joint": 0.0, #-0.8757, 0
                "right_inner_finger_joint": 0.0, #-0.8757, 0
            },            
        ),
        actuators={
            "kinova_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["gen3_joint_[1-4]"],
                effort_limit=5000.0,
                velocity_limit=2.0,
                stiffness=800.0,
                damping=160.0,
            ),
            "kinova_forearm": ImplicitActuatorCfg(
                joint_names_expr=["gen3_joint_[5-7]"],
                effort_limit=5000.0,
                velocity_limit=2.5,
                stiffness=800.0,
                damping=160.0,
            ),
            "kinova_gripper": ImplicitActuatorCfg(
                joint_names_expr=['finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint'],
                effort_limit=200.0,
                velocity_limit=5.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )

    # cube_cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,  # Enable gravity for more realistic physics
    #         max_depenetration_velocity=5.0,
    #         solver_position_iteration_count=16,  # Better collision resolution
    #         solver_velocity_iteration_count=1,
    #         max_angular_velocity=1000.0,
    #         max_linear_velocity=1000.0,
    #     ),
    #     scale=(0.06, 0.06, 0.06),

    # )
    
    cube_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # Enable gravity for more realistic physics
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=16,  # Better collision resolution
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                ),
            scale=(1.0, 0.7, 0.8),

            ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.67, 0.0, 0.03),
            rot=(1.0, 0.0, 0.0, 0.0)
        )
        )

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    # Additional parameters
    action_scale = 1.0


class RobotEnvDiffIK(DirectRLEnv):
    cfg = RobotEnvCfgDiffIK()

    def __init__(self, cfg:RobotEnvCfgDiffIK, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.rl_dt = self.physics_dt*self.cfg.decimation

        ## setting the controller
        self.diff_ik_controller = DifferentialIKController(self.cfg.diff_ik_cfg, self.num_envs, device=self.device)


        ## bodies and joints of the robot

        # print(self._robot.body_names)
        # print(self._robot.joint_names)
        # ['world', 'gen3_base_link', 'gen3_shoulder_link', 'gen3_half_arm_1_link', 'gen3_half_arm_2_link', 'gen3_forearm_link', 'gen3_spherical_wrist_1_link', 'gen3_spherical_wrist_2_link', 'gen3_bracelet_link', 'gen3_end_effector_link', 'gripper_end_effector_link', 'robotiq_arg2f_base_link', 'left_outer_knuckle', 'left_inner_knuckle', 'right_inner_knuckle', 'right_outer_knuckle', 'left_outer_finger', 'right_outer_finger', 'left_inner_finger', 'right_inner_finger', 'left_inner_finger_pad', 'right_inner_finger_pad']
        # ['gen3_joint_1', 'gen3_joint_2', 'gen3_joint_3', 'gen3_joint_4', 'gen3_joint_5', 'gen3_joint_6', 'gen3_joint_7', 'finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint']

        ef_body_name = "gen3_end_effector_link"
        arm_joint_names = ["gen3_joint_.*"]

        self.ef_body_idx = self._robot.find_bodies(ef_body_name)[0][0]
        self.arm_joint_ids = self._robot.find_joints(arm_joint_names)[0]
        
        self.ee_jacobi_idx = self.ef_body_idx
        
        # Gripper
        self.gripper_joint_names = ['finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 
                                  'right_outer_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint']
        self.gripper_joint_ids = self._robot.find_joints(self.gripper_joint_names)[0]
        self.gripper_open_val = torch.tensor([0.1], device=self.device)
        self.gripper_close_val = torch.tensor([0.8], device=self.device)
        self.gripper_multiplier = torch.tensor([[1, 1.0, 1.0, 1, -0.8, -0.8]], device=self.device)


        ## friction
        ## intertia tensors
        ## intiial tensors
        ## gripper

        ## logging data after every physics step
        

    def _setup_scene(self):
        super()._setup_scene()
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        table_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        table_cfg.func(
            "/World/envs/env_.*/Table", table_cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        # Spawn a cube into the scene

        # spawn_from_usd(
        #     prim_path = "/World/envs/env_.*/Cube",
        #     cfg=self.cfg.cube_cfg,
        #     translation=(0.7, 0.0, 0.75),
        #     orientation=(0.0, 0.0, 0.0, 1.0),
        # )

        self._cube = RigidObject(self.cfg.cube_cfg)

        self.scene.rigid_objects["cube"] = self._cube

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)
        spawn_light(prim_path="/World/Light", cfg=light_cfg)

        self.scene.clone_environments(copy_from_source=False) ## if set to true we can have independent usd prims => independet robot cfgs, other assets

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()
        self._compute_intermediate_values(self.physics_dt)
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        delta_ee_pos = self.actions[:,:6]        
        self.gripper_action()
        self.diff_ik_controller.set_command(delta_ee_pos, ee_pos_b, ee_quat_b)
        
    def _apply_action(self):

        self._compute_intermediate_values(self.physics_dt)
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        jacobian_b = self._compute_frame_jacobian()
        target_joint_pos = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, self.joint_pos[:, self.arm_joint_ids])
        self._robot.set_joint_position_target(target_joint_pos, self.arm_joint_ids)


    def _get_rewards(self):
        rewards = torch.zeros(self.num_envs, device = self.device)
        return rewards

    def _get_observations(self):
        # observations = torch.zeros((self.num_envs, ), device=self.device)
        pass
        # return observations

    def _get_dones(self):
        
        truncated = self.episode_length_buf >= self.max_episode_length-1
        return truncated, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids = env_ids)
        self._robot.set_joint_velocity_target(joint_vel, env_ids = env_ids)

        self.step_sim_no_action()
    
    def step_sim_no_action(self):
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(self.physics_dt)

    def _compute_intermediate_values(self, dt):
        self.current_ef_pos = self._robot.data.body_pos_w[:, self.ef_body_idx].clone()
        self.current_ef_quat = self._robot.data.body_quat_w[:, self.ef_body_idx].clone()
        self.joint_pos = self._robot.data.joint_pos.clone()
        pass

    def _compute_frame_pose(self):
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._robot.data.body_pos_w[:, self.ef_body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self.ef_body_idx]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        return ee_pose_b, ee_quat_b
        pass

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, 0:7]
        base_rot = self._robot.data.root_quat_w
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        
        return jacobian 

    def gripper_action(self, env_ids=None):

        gripper_command = self.actions[:, -1].clone()
        gripper_command = gripper_command.reshape(-1,1)
        gripper_pos = torch.where(gripper_command > 0, self.gripper_close_val, self.gripper_open_val)

        # print(gripper_pos, gripper_pos.shape)
        # print(self.gripper_multiplier, self.gripper_multiplier.shape)

        gripper_pos = gripper_pos*self.gripper_multiplier
        # print("gripper joints \n", gripper_pos, gripper_pos.shape)
        self._robot.set_joint_position_target(gripper_pos, self.gripper_joint_ids, env_ids=env_ids)



class RobotEnvLogging(RobotEnvDiffIK):
    def __init__(self, cfg:RobotEnvCfgDiffIK, render_mode: str | None = None, log_file=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize logger
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = "scripts/custom_scripts/logs/csv_files"
            os.makedirs(log_dir, exist_ok=True)
            log_file = f"{log_dir}/robot_physics_data_{timestamp}.csv"
        
        ## deleting the logfile if it exists
        if os.path.exists(log_file):
            os.remove(log_file)


        self.log_file = log_file
        self.log_data = []
        self.physics_step_count = 0

        self.recording_state = False

    # def _pre_physics_step(self, actions):
    #     super()._pre_physics_step(actions)
        
    #     return 

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        
        self.log_data = []
        self.physics_step_count = 0
        
    def _apply_action(self):
        super()._apply_action()
        if self.recording_state:
            self._log_physics_step()

    def _log_physics_step(self):
        action_np = self.actions.clone().cpu().numpy()
        env_idx = 0
        log_entry = {
            'physics_step': self.physics_step_count,
            'env_id': env_idx,
            'ee_pos_x': self.current_ef_pos[env_idx, 0].item(),
            'ee_pos_y': self.current_ef_pos[env_idx, 1].item(),
            'ee_pos_z': self.current_ef_pos[env_idx, 2].item(),
            'ee_quat_w': self.current_ef_quat[env_idx, 0].item(),
            'ee_quat_x': self.current_ef_quat[env_idx, 1].item(),
            'ee_quat_y': self.current_ef_quat[env_idx, 2].item(),
            'ee_quat_z': self.current_ef_quat[env_idx, 3].item(),
        }
        
        # Add actions
        for i in range(7):
            log_entry[f'action_{i}'] = action_np[env_idx, i] if i < action_np.shape[1] else 0.0
        
        # Add joint positions
        for i, joint_idx in enumerate(self.arm_joint_ids):
            log_entry[f'arm_joint_{i}'] = self.joint_pos[env_idx, joint_idx].item()

        self.log_data.append(log_entry)

        self.physics_step_count += 1
        
        # Save periodically
        if self.physics_step_count % 100 == 0:
            self.save_to_csv()
    
    def save_to_csv(self):
        if self.log_data:
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.log_file, index=False)
            print(f"Saved {len(self.log_data)} physics step entries to {self.log_file}")
    
    def close(self):
        """Override close to save final data"""
        self.save_to_csv()
        super().close() if hasattr(super(), 'close') else None



## rewards, observations, dones, actions, reset

