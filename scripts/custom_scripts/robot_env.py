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


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "usd")

@configclass
class RobotEnvCfg(DirectRLEnvCfg):

    episode_length_s = 20 ## 20sec = (rl_steps)*decimation*step_dt ## naming shouldnt be changed
    decimation = 20  ## decimation 20 means after every 20 sim steps one rl steps
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
            joint_pos={
                "gen3_joint_1": -0.0390,
                "gen3_joint_2": 0.8417,
                "gen3_joint_3": -0.0531,
                "gen3_joint_4": 2.2894,
                "gen3_joint_5": -0.0744,
                "gen3_joint_6": -1.5667,
                "gen3_joint_7": -1.5310,
                "finger_joint": 0.0, # 0, 0.8
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
                effort_limit=80.0,
                velocity_limit=2.0,
                stiffness=800.0,
                damping=160.0,
            ),
            "kinova_forearm": ImplicitActuatorCfg(
                joint_names_expr=["gen3_joint_[5-7]"],
                effort_limit=10.0,
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


class RobotEnv(DirectRLEnv):
    cfg = RobotEnvCfg()

    def __init__(self, cfg:RobotEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.gripper_multiplier = torch.tensor([1, 1.0, 1.0, 1, -0.8, -0.8], device=self.device)


        ## friction
        ## intertia tensors
        ## intiial tensors
        ## gripper

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
        
        self.diff_ik_controller.set_command(delta_ee_pos, ee_pos_b, ee_quat_b)
        
    def _apply_action(self):

        self._compute_intermediate_values(self.physics_dt)
        jacobian_b = self._compute_frame_jacobian()
        target_joint_pos = self.diff_ik_controller.compute(self.current_ef_pos, self.current_ef_quat, jacobian_b, self.joint_pos)
        self._robot.set_joint_position_target()

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

    def gripper_action(self, env_ids):

        gripper_command = self.actions[:, 6]
        gripper_pos = torch.where(gripper_command > 0, self.gripper_close_val, self.gripper_open_val)*self.gripper_multiplier
        self._robot.set_joint_position_target(gripper_pos, self.gripper_joint_ids, env_ids=env_ids)
        


## rewards, observations, dones, actions, reset

