#!/usr/bin/env python3
"""
Simple visualization script for comparing joint angle step responses across real, rviz, and isaac platforms.
Uses rerun.io to visualize current vs desired joint angles from logged CSV and JSON data.
"""

import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import rerun as rr


def parse_folder_name(folder_name: str) -> dict:
    """
    Parse folder name to extract joint index, platform type, and experiment number.
    Expected format: {joint_idx}_joint_{type}_{exp_no}
    Example: 0_joint_rviz_1 -> {'joint_idx': 0, 'type': 'rviz', 'exp_no': 1}
    """
    pattern = r"(\d+)_joint_(rviz|real|isaac)_(\d+)"
    match = re.match(pattern, folder_name)
    if match:
        return {
            'joint_idx': int(match.group(1)),
            'type': match.group(2),
            'exp_no': int(match.group(3))
        }
    return None


def main():
    # Initialize rerun
    rr.init("joint_step_response_comparison", spawn=True)

    # Base directory containing all experiment logs
    base_dir = Path("/home/tih_auto_hpz4/Aditya/IsaacLab/scripts/peg_hole_2/joint_profiling_logs")

    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return

    # Iterate through all subdirectories
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        # Parse folder name
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            print(f"Skipping folder {folder.name} - doesn't match expected naming pattern")
            continue

        joint_idx = parsed['joint_idx']
        platform_type = parsed['type']
        exp_no = parsed['exp_no']

        # Check for required files
        csv_path = folder / "step_data.csv"
        json_path = folder / "metadata.json"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {folder.name}")
            continue
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping {folder.name}")
            continue

        print(f"Processing: {folder.name}")

        # Read metadata to get desired joint angle
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Get desired angle in degrees for this specific joint
        desired_angles_deg = metadata.get('desired_joint_angles_deg', [])
        if joint_idx >= len(desired_angles_deg):
            print(f"Warning: joint_idx {joint_idx} not found in metadata, skipping")
            continue

        desired_angle = desired_angles_deg[joint_idx]

        # Read CSV data
        df = pd.read_csv(csv_path)

        # Extract step count and current joint position for the specific joint
        steps = df['recording_step_count'].values.astype(np.int64)
        current_joint_col = f'current_joint_{joint_idx}'

        if current_joint_col not in df.columns:
            print(f"Warning: {current_joint_col} not found in CSV, skipping")
            continue

        # Convert current angles from radians to degrees
        current_angles_rad = df[current_joint_col].values.astype(np.float32)
        current_angles = np.degrees(current_angles_rad)

        # Create constant array for desired angle
        desired_angles_array = np.full_like(current_angles, desired_angle, dtype=np.float32)

        # Define entity paths
        exp_name = f"{platform_type}_{exp_no}"
        current_path = f"/{joint_idx}/{exp_name}/current"
        desired_path = f"/{joint_idx}/{exp_name}/desired"

        # Log current joint angles
        rr.log(desired_path, rr.SeriesLines(widths=1), static=True)
        rr.send_columns(
            current_path,
            indexes=[rr.TimeColumn("step", sequence=steps)],
            columns=rr.Scalars.columns(scalars=current_angles)
        )

        # Log desired joint angles (constant) with thicker line
        rr.log(desired_path, rr.SeriesLines(widths=1.5, colors=[255,0,0]), static=True)
        rr.send_columns(
            desired_path,
            indexes=[rr.TimeColumn("step", sequence=steps)],
            columns=rr.Scalars.columns(scalars=desired_angles_array)
        )

        print(f"  Logged {len(steps)} steps for joint {joint_idx} ({exp_name})")

    print("\nVisualization complete! Check the rerun viewer.")


if __name__ == "__main__":
    main()
