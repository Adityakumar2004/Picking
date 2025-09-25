import rerun as rr
import pandas as pd
import numpy as np
from pathlib import Path


def visualize_robot_data(csv_path, exp_name):
    # Initialize rerun
    rr.init(exp_name)
    rr.spawn()
    
    # Read CSV data
    df = pd.read_csv(csv_path)

    # Use physics_step as the timeline index
    trajectory_points = []
    desired_trajectory = []
    initial_pos = [df.iloc[0]['ee_pos_x'], df.iloc[0]['ee_pos_y'], df.iloc[0]['ee_pos_z']]
    index = 0
    for _, row in df.iterrows():
        physics_step = int(row['physics_step'])

        # Set timeline for this physics step
        rr.set_time(timeline="physics_step", sequence=physics_step)

        # Log end-effector position
        ee_pos = [row['ee_pos_x'], row['ee_pos_y'], row['ee_pos_z']]
        print(ee_pos)
        rr.log(f"{exp_name}/robot/end_effector/position", rr.Points3D([ee_pos], colors=[255, 0, 0], radii=0.005))
        trajectory_points.append(ee_pos)
        # Logging the trajectory of the end-effector
        rr.log(f"{exp_name}/robot/end_effector/trajectory_followed", rr.LineStrips3D(trajectory_points))

        # Log end-effector orientation (quaternion)
        ee_quat = [row['ee_quat_x'], row['ee_quat_y'], row['ee_quat_z'], row['ee_quat_w']]
        rr.log(f"{exp_name}/robot/end_effector/orientation", rr.Transform3D(translation=ee_pos, rotation=rr.Quaternion(xyzw=ee_quat)))

        # Log joint positions as a scalar plot
        joint_positions = [row[f'arm_joint_{i}'] for i in range(7)]
        for i, joint_pos in enumerate(joint_positions):
            rr.log(f"{exp_name}/robot/joints/joint_{i}", rr.Scalars(joint_pos))

        # Log actions as scalars
        # actions = [row[f'action_{i}'] for i in range(7)]
        # for i, action in enumerate(actions):
        #     rr.log(f"robot/actions/action_{i}", rr.Scalar(action))

        ## logging desired positon (actions + current pos)
        delta_pos = [row[f'action_{i}'] for i in range(3)]

        desired_pos = np.array(desired_trajectory[-1]) if len(desired_trajectory) > 0 else np.array(initial_pos)
        if index % 20 == 0:  
            print(index)
            desired_pos = desired_pos + np.array(delta_pos)


        desired_trajectory.append(desired_pos.tolist())
        rr.log(f"{exp_name}/robot/end_effector/desired_trajectory", rr.LineStrips3D([desired_trajectory], colors=[0, 255, 0]))
        rr.log(f"{exp_name}/robot/end_effector/desired_position", rr.Points3D([desired_pos.tolist()], colors=[0, 255, 0], radii=0.005))

        index += 1


if __name__ == "__main__":
    exp_name = "test_log"
    log_dir = "scripts/custom_scripts/logs/csv_files"
    csv_file = f"{log_dir}/{exp_name}.csv"
    visualize_robot_data(csv_file, exp_name)
