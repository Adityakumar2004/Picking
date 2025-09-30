import rerun as rr
import pandas as pd
import time
import glob
import os
import numpy as np

# Folder where experiments drop CSVs
LOG_DIR = "scripts/custom_scripts/logs/csv_files"
# Address of Rerun Viewer (start with: rerun)
# VIEWER_ADDR = "127.0.0.1:9876"
# VIEWER_ADDR = "grpc://127.0.0.1:9876"
VIEWER_ADDR = "grpc://127.0.0.1:9876"


# Start rerun logger
rr.init("csv_pipeline", spawn=True)
# rr.connect_grpc(VIEWER_ADDR)
# Keep track of processed files
seen = set()
decimation = 20
while True:
    for file in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        if file not in seen:
            try:
                df = pd.read_csv(file)
                trajectory_points = []
                desired_trajectory = []
                index = 0
                initial_pos = [df.iloc[0]['ee_pos_x'], df.iloc[0]['ee_pos_y'], df.iloc[0]['ee_pos_z']]
                exp_name = os.path.splitext(os.path.basename(file))[0]  # e.g. "exp1"

                # Assumes CSV has a "step" column
                if "physics_step" not in df.columns:
                    print(f"Skipping {file}: no 'physics_step' column")
                    continue

                steps = df["physics_step"].to_numpy()

                # Log all other columns as time series
                # for col in df.columns:
                #     if col == "physics_step":
                #         continue
                #     rr.TimeColumn("physics_step",sequence= steps)
                #     rr.log(f"{exp_name}/{col}", rr.Scalars(df[col].to_numpy()))

                # for col in df.columns:
                #     if col == "physics_step":
                #         continue
                #     scalars = df[col].to_numpy()
                #     rr.send_columns(
                #     f"scalars/{exp_name}/{col}",
                #     indexes=[rr.TimeColumn("step", sequence=steps)],
                #     columns=rr.Scalars.columns(scalars=scalars),
                #     )

                ee_pos_list = ["ee_pos_x", "ee_pos_y", "ee_pos_z"]

                for col in ee_pos_list:
                    rr.send_columns(
                        f"scalars/{col}/{exp_name}",
                        indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                        columns=rr.Scalars.columns(scalars=df[col].to_numpy()),
                    )

                    rr.send_columns(
                        f"ef_pos/{exp_name}",
                        indexes=[rr.TimeColumn("physics_step", sequence=steps)],
                        columns=rr.Points3D.columns(positions=df[ee_pos_list].to_numpy()),
                    )
                
                for _, row in df.iterrows():
                    physics_step = int(row['physics_step'])
                    ee_pos = [row['ee_pos_x'], row['ee_pos_y'], row['ee_pos_z']]
                    trajectory_points.append(ee_pos)
                    # print(ee_pos)
                    rr.set_time(timeline="physics_step", sequence=physics_step)
                    rr.log(f"ef_trajectory/{exp_name}", rr.LineStrips3D(trajectory_points))


                    ## logging desired positon (actions + current pos)
                    delta_pos = [row[f'action_{i}'] for i in range(3)]

                    desired_pos = np.array(desired_trajectory[-1]) if len(desired_trajectory) > 0 else np.array(initial_pos)
                    if index % decimation == 0:  
                        # print(index)
                        desired_pos = desired_pos + np.array(delta_pos)


                    desired_trajectory.append(desired_pos.tolist())
                    rr.log(f"desired_trajectory/{exp_name}", rr.LineStrips3D([desired_trajectory], colors=[0, 255, 0]))

                    rr.log(f"scalars/desired_pos_x/{exp_name}", rr.Scalars(desired_pos[0]))
                    rr.log(f"scalars/desired_pos_y/{exp_name}", rr.Scalars(desired_pos[1]))
                    rr.log(f"scalars/desired_pos_z/{exp_name}", rr.Scalars(desired_pos[2]))

                    index += 1
                
                # print(trajectory_points)
                print(f"Logged {file} into Rerun")
                seen.add(file)

            except Exception as e:
                print(f"Error loading {file}: {e}")

    # time.sleep(2)  # poll every 2s for new files
