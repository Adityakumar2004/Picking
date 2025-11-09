import json
import rerun as rr
import pandas as pd
import time
import glob
import os
import numpy as np

PARENT_LOG_DIR = "scripts/peg_hole_2/log_profiling_2"
VIEWER_ADDR = "grpc://127.0.0.1:9876"

rr.init("robot_profiling", spawn=True)
# rr.spawn()
seen = set()
# for folder in os.listdir(PARENT_LOG_DIR):
#     print(f"Processing folder using os: {folder}")

for folder in glob.glob(os.path.join(PARENT_LOG_DIR, "*")):
    # print(f"Processing folder using glob: {folder}") 
    if folder not in seen:
        ## check the basename of folder to avoid *rviz* folders
        try:
            if "rviz" not in os.path.basename(folder) and "real" not in os.path.basename(folder):
                ## reading the json file for parameters of the experiment
                with open(os.path.join(folder, "config.json"), 'r') as f:
                    config_data = json.load(f)
                
                df = pd.read_csv(os.path.join(folder, "rl_step_data.csv"))
                steps = df["rl_step"].to_numpy(dtype=np.int32)
                exp_name = os.path.basename(folder)
                # print(f"Experiment Name: {exp_name}")

                ee_pos_list_current = ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
                ee_pos_list_desired = ["desired_ee_pos_x", "desired_ee_pos_y", "desired_ee_pos_z"]    
                for axis in ['x', 'y', 'z']:
                    
                    rr.send_columns(f"pos_graphs/{axis}/current_{axis}/{exp_name}",
                                    indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                    columns=rr.Scalars.columns(scalars=df[f"ee_pos_{axis}"].to_numpy())
                                    )
                    rr.send_columns(f"pos_graphs/{axis}/desired_{axis}/{exp_name}",
                                    indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                    columns=rr.Scalars.columns(scalars=df[f"desired_ee_pos_{axis}"].to_numpy()),
                                    )
                    rr.send_columns(f"vel_graphs/{axis}/{exp_name}",
                                    indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                    columns=rr.Scalars.columns(scalars=df[f"ee_lin_vel_{axis}"].to_numpy())
                                    )
                ## adding joint velocities, torques and their limits
                for i in range(7):
                    rr.send_columns(f"joint_velocities/joint_{i}/{exp_name}",
                                    indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                    columns=rr.Scalars.columns(scalars=df[f"arm_joint_vel_{i}"].to_numpy())
                                    )
                    rr.send_columns(f"joint_torques/joint_{i}/{exp_name}",
                                    indexes=[rr.TimeColumn("physics_step", sequence=steps)],
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
                                indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                columns=rr.Scalars.columns(scalars=np.full_like(steps, vel_limit, dtype=np.float32))
                                )
                rr.send_columns(f"joint_velocities/vel_limit/-/{exp_name}",
                                indexes=[rr.TimeColumn("physics_step", sequence=steps)],
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
                                indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                columns=rr.Scalars.columns(scalars=np.full_like(steps, torque_limit, dtype=np.float32))
                                )
                rr.send_columns(f"joint_torques/torque_limit/-/{exp_name}",
                                indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                columns=rr.Scalars.columns(scalars=np.full_like(steps, -torque_limit, dtype=np.float32))
                                )

                    
                seen.add(folder)

                trajectory_points = []
                desired_trajectory = []
                for _,row in df.iterrows():
                    rl_step = int(row["rl_step"])
                    rr.set_time(timeline = "physics_step", sequence= rl_step)

                    ee_pos = np.array([row["ee_pos_x"], row["ee_pos_y"], row["ee_pos_z"]])
                    trajectory_points.append(ee_pos)
                    rr.log(f"trajectory/{exp_name}", rr.LineStrips3D(trajectory_points))

            else:
                df = pd.read_csv(os.path.join(folder, "step_data.csv"))
                steps = df["physics_step"].to_numpy(dtype=np.int32)
                exp_name = os.path.basename(folder)
                # print(f"Experiment Name: {exp_name}")

                ee_pos_list_current = ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
                ee_pos_list_desired = ["desired_ee_pos_x", "desired_ee_pos_y", "desired_ee_pos_z"]    
                for axis in ['x', 'y', 'z']:
                    
                    rr.send_columns(f"pos_graphs/{axis}/current_{axis}/{exp_name}",
                                    indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                    columns=rr.Scalars.columns(scalars=df[f"ee_pos_{axis}"].to_numpy())
                                    )
                    rr.send_columns(f"pos_graphs/{axis}/desired_{axis}/{exp_name}",
                                    indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                    columns=rr.Scalars.columns(scalars=df[f"desired_ee_pos_{axis}"].to_numpy()),
                                    )
                    
                    rr.send_columns(f"vel_graphs/{axis}/{exp_name}",
                                    indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                    columns=rr.Scalars.columns(scalars=df[f"ee_lin_vel_{axis}"].to_numpy())
                                    )
                            
                trajectory_points = []
                desired_trajectory = []
                for _,row in df.iterrows():
                    rl_step = int(row["physics_step"])
                    rr.set_time(timeline = "physics_step", sequence= rl_step)

                    ee_pos = np.array([row["ee_pos_x"], row["ee_pos_y"], row["ee_pos_z"]])
                    trajectory_points.append(ee_pos)
                    rr.log(f"trajectory/{exp_name}", rr.LineStrips3D(trajectory_points))

                if os.path.isfile(os.path.join(folder, "key_points.csv")):
                    df_key = pd.read_csv(os.path.join(folder, "key_points.csv"))
                    steps = df_key["physics_step"].to_numpy(dtype=np.int32)
                    for axis in ['x', 'y', 'z']:
                        rr.send_columns(f"key_points_graphs/{axis}/{exp_name}",
                                        indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                                        columns=rr.Scalars.columns(scalars=df_key[f"ee_pos_{axis}"].to_numpy())
                                        )

                    # points = df_key[["ee_pos_x", "ee_pos_y", "ee_pos_z"]].to_numpy().repeat(2,axis = 0)
                    
                    pos_list = []
                    for _,row in df_key.iterrows():
                        physics_step = int(row["physics_step"])
                        rr.set_time(timeline = "physics_step", sequence= physics_step)
                        point = row[["ee_pos_x", "ee_pos_y", "ee_pos_z"]].to_list()
                        pos_list.append(point)
                        rr.log(f"key_points/{exp_name}", rr.Points3D(positions=pos_list, radii=0.0007))
                    
                    ## taking the average, variance of the alternate clusters of keypoints
                    key_points_array = df_key[["ee_pos_x", "ee_pos_y", "ee_pos_z"]].to_numpy()
                    cluster_1 = key_points_array[::2]
                    cluster_2 = key_points_array[1::2]
                    mean_cluster_1 = np.mean(cluster_1, axis=0)
                    mean_cluster_2 = np.mean(cluster_2, axis=0)
                    var_cluster_1 = np.var(cluster_1, axis=0)
                    var_cluster_2 = np.var(cluster_2, axis=0)
                    print("----------------------------------")
                    print("file :", folder)
                    print(f"Mean Cluster 1: {mean_cluster_1}, Variance Cluster 1: {var_cluster_1}")
                    print(f"Mean Cluster 2: {mean_cluster_2}, Variance Cluster 2: {var_cluster_2}")
                    ## printing the desired positions taken from df at these keypoints taken from df_key
                    print("Desired positions at key points:")
                    ## getting the index of physics steps in df corresponding to physics steps in df_key
                    key_time = df_key["physics_step"].to_list()
                    t1 = key_time[1]
                    t2 = key_time[2]
                    desired_pos_t1 = df[df["physics_step"] == t1][["desired_ee_pos_x", "desired_ee_pos_y", "desired_ee_pos_z"]].to_numpy()
                    desired_pos_t2 = df[df["physics_step"] == t2][["desired_ee_pos_x", "desired_ee_pos_y", "desired_ee_pos_z"]].to_numpy()
                    print(f"desired pos at time {t1}: {desired_pos_t1}")
                    print(f"desired pos at time {t2}: {desired_pos_t2}")



                seen.add(folder)


        except Exception as e:
            print(f"Error loading {folder}: {e}")

