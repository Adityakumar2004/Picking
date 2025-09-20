
import math
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space error between target Franka fingertip pose and current pose."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf

    # Compute pos error
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # Compute rot error
    if jacobian_type == "geometric":  # See example 2.9.8; note use of J_g and transformation between rotation vectors
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html

        # Check for shortest path using quaternion dot product.
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[
            :, 0
        ]  # scalar component
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def compute_ik_diff_dof_torque(
        curr_fingertip_midpoint_pos, 
        curr_fingertip_midpoint_quat, 
        target_midpoint_pos, 
        target_midpoint_quat, 
        curr_joint_pos, 
        jacobian, 
        diff_ik_controller:DifferentialIKController,
        cfg, 
        device):

    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, curr_joint_pos.shape[1]), device=device)
    target_joint_pos = torch.zeros((num_envs, curr_joint_pos.shape[1]), device=device)
    pos_error, axis_angle_error = get_pose_error(curr_fingertip_midpoint_pos, curr_fingertip_midpoint_quat, target_midpoint_pos, target_midpoint_quat, 'geometric', 'axis_angle')
    
    ee_pose_error = torch.cat((pos_error, axis_angle_error), dim=1)

    # print("ee pose error: \n", ee_pose_error)
    
    diff_ik_controller.set_command(command=ee_pose_error, ee_pos=curr_fingertip_midpoint_pos, ee_quat=curr_fingertip_midpoint_quat)

    target_joint_pos[:,:7] = diff_ik_controller.compute(curr_fingertip_midpoint_pos, curr_fingertip_midpoint_quat, jacobian, curr_joint_pos[:, :7].clone())

    target_joint_pos[:, 7:] = curr_joint_pos[:, 7:]

    dof_torque[:, 0:7] = apply_dof_gains(kp = 100, target_joint_pos=target_joint_pos, curr_joint_pos=curr_joint_pos, num_envs=num_envs, device=device)
    # print("i am here")
    return dof_torque, target_joint_pos


def apply_dof_gains(kp, target_joint_pos, curr_joint_pos, num_envs, device):
    torque = torch.zeros((num_envs, curr_joint_pos.shape[1]), device=device)
    torque = kp * (target_joint_pos - curr_joint_pos)[:, :7]
    return torque