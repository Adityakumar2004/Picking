import json
import rerun as rr
import pandas as pd
import time
import glob
import os
import numpy as np

PARENT_LOG_DIR = "scripts/peg_hole_2/log_profiling"
VIEWER_ADDR = "grpc://127.0.0.1:9876"

rr.init("robot_profiling", spawn=True)
# rr.spawn()
seen = set()
# for folder in os.listdir(PARENT_LOG_DIR):
#     print(f"Processing folder using os: {folder}")

for folder in glob.glob(os.path.join(PARENT_LOG_DIR, "*")):
    # print(f"Processing folder using glob: {folder}") 
    if folder not in seen:
        try:
            ## reading the json file for parameters of the experiment
            with open(os.path.join(folder, "config.json"), 'r') as f:
                config_data = json.load(f)
            
            df = pd.read_csv(os.path.join(folder, "rl_step_data.csv"))
            steps = df["rl_step"].to_numpy(dtype=np.int32)
            exp_name = os.path.basename(folder)
            print(f"Experiment Name: {exp_name}")

            ee_pos_list_current = ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
            ee_pos_list_desired = ["desired_ee_pos_x", "desired_ee_pos_y", "desired_ee_pos_z"]    
            for axis in ['x', 'y', 'z']:
                
                rr.send_columns(f"pos_graphs/{axis}/current_{axis}/{exp_name}",
                                indexes=[rr.TimeColumn("rl_step", sequence=steps)],
                                columns=rr.Scalars.columns(scalars=df[f"ee_pos_{axis}"].to_numpy())
                                )
                rr.send_columns(f"pos_graphs/{axis}/desired_{axis}/{exp_name}",
                                indexes=[rr.TimeColumn("rl_step", sequence=steps)],
                                columns=rr.Scalars.columns(scalars=df[f"desired_ee_pos_{axis}"].to_numpy()),
                                )
            
            ## adding joint velocities, torques and their limits
            for i in range(7):
                rr.send_columns(f"joint_velocities/joint_{i}/{exp_name}",
                                indexes=[rr.TimeColumn("rl_step", sequence=steps)],
                                columns=rr.Scalars.columns(scalars=df[f"arm_joint_vel_{i}"].to_numpy())
                                )
                rr.send_columns(f"joint_torques/joint_{i}/{exp_name}",
                                indexes=[rr.TimeColumn("rl_step", sequence=steps)],
                                columns=rr.Scalars.columns(scalars=df[f"arm_joint_torque_{i}"].to_numpy())
                                )

            vel_limit = config_data.get("velocity_limits", 0)
            # Log velocity limits with distinct styling (red color, thinner line)
            rr.log(f"joint_velocities/vel_limit/+/{exp_name}", rr.SeriesLines(
                colors=[255, 0, 0],  # Red color for limits
                widths=1.5,
                names=f"vel_limit"
            ), static=True)
            rr.send_columns(f"joint_velocities/vel_limit/+/{exp_name}",
                            indexes=[rr.TimeColumn("rl_step", sequence=steps)],
                            columns=rr.Scalars.columns(scalars=np.full_like(steps, vel_limit, dtype=np.float32))
                            )
            rr.send_columns(f"joint_velocities/vel_limit/-/{exp_name}",
                            indexes=[rr.TimeColumn("rl_step", sequence=steps)],
                            columns=rr.Scalars.columns(scalars=np.full_like(steps, -vel_limit, dtype=np.float32))
                            )

            torque_limit = config_data.get("torque_limits", 0)
            # Log torque limits with distinct styling (red color, thinner line)
            rr.log(f"joint_torques/torque_limit/+/{exp_name}", rr.SeriesLines(
                colors=[255, 0, 0],  # Red color for limits
                widths=1.5,
                names=f"torque_limit"
            ), static=True)
            rr.send_columns(f"joint_torques/torque_limit/+/{exp_name}",
                            indexes=[rr.TimeColumn("rl_step", sequence=steps)],
                            columns=rr.Scalars.columns(scalars=np.full_like(steps, torque_limit, dtype=np.float32))
                            )
            rr.send_columns(f"joint_torques/torque_limit/-/{exp_name}",
                            indexes=[rr.TimeColumn("rl_step", sequence=steps)],
                            columns=rr.Scalars.columns(scalars=np.full_like(steps, -torque_limit, dtype=np.float32))
                            )

                
            seen.add(folder)

            trajectory_points = []
            desired_trajectory = []
            for _,row in df.iterrows():
                rl_step = int(row["rl_step"])
                rr.set_time(timeline = "rl_step", sequence= rl_step)

                ee_pos = np.array([row["ee_pos_x"], row["ee_pos_y"], row["ee_pos_z"]])
                trajectory_points.append(ee_pos)
                rr.log(f"trajectory/{exp_name}", rr.LineStrips3D(trajectory_points))

            
            


        except Exception as e:
            print(f"Error loading {folder}: {e}")