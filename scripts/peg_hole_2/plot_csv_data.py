#!/usr/bin/env python3
"""
Script to plot CSV data with individual plots for each column.
Saves plots in a subfolder called 'plots'.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_csv_data(csv_path: str, output_dir: str):
    """
    Read CSV file and create individual plots for each column.

    Args:
        csv_path: Path to the CSV file
        output_dir: Directory where plots subfolder will be created
    """
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Found {len(df.columns)} columns and {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Create output directory for plots
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    # Get step column (x-axis)
    if 'step' in df.columns:
        x_data = df['step'].values
        x_label = 'Step'
        columns_to_plot = [col for col in df.columns if col != 'step']
    else:
        x_data = np.arange(len(df))
        x_label = 'Index'
        columns_to_plot = df.columns

    # Calculate desired positions by accumulating processed actions
    desired_positions = {}
    if 'fingertip_midpoint_pos_0' in df.columns and 'processed_action_0' in df.columns:
        # Get initial positions
        initial_pos_x = df['fingertip_midpoint_pos_0'].iloc[0]
        initial_pos_y = df['fingertip_midpoint_pos_1'].iloc[0]
        initial_pos_z = df['fingertip_midpoint_pos_2'].iloc[0]

        # Calculate desired positions by accumulating actions
        desired_positions['fingertip_midpoint_pos_0'] = initial_pos_x + df['processed_action_0'].cumsum().values
        desired_positions['fingertip_midpoint_pos_1'] = initial_pos_y + df['processed_action_1'].cumsum().values
        desired_positions['fingertip_midpoint_pos_2'] = initial_pos_z + df['processed_action_2'].cumsum().values

        print(f"\nCalculated desired positions from processed actions")

    # Create individual plot for each column
    for col in columns_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))

        y_data = df[col].values

        # Check if this is a position column and we have desired position data
        has_desired = col in desired_positions

        if has_desired:
            # Plot actual position
            ax.plot(x_data, y_data, 'b-', linewidth=1.5, alpha=0.7, label=f'{col} (Actual)')
            ax.plot(x_data, y_data, 'bo', markersize=3, alpha=0.6)

            # Plot desired position
            desired_y = desired_positions[col]
            ax.plot(x_data, desired_y, 'r--', linewidth=1.5, alpha=0.7, label=f'{col} (Desired)')
            ax.plot(x_data, desired_y, 'ro', markersize=3, alpha=0.6)

            # Calculate error
            error = y_data - desired_y
            error_mean = np.mean(np.abs(error))
            error_std = np.std(error)
        else:
            # Plot line with markers (original behavior)
            ax.plot(x_data, y_data, 'b-', linewidth=1.5, alpha=0.7, label=col)
            ax.plot(x_data, y_data, 'ro', markersize=3, alpha=0.6)

        # Labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        title = f'{col} vs {x_label}'
        if has_desired:
            title += ' (Actual vs Desired)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics in text box
        stats_text = f'Actual Min: {y_data.min():.4f}\n'
        stats_text += f'Actual Max: {y_data.max():.4f}\n'
        stats_text += f'Actual Mean: {y_data.mean():.4f}\n'
        stats_text += f'Actual Std: {y_data.std():.4f}'

        if has_desired:
            stats_text += f'\n\nTracking Error:\n'
            stats_text += f'Mean |Error|: {error_mean:.4f}\n'
            stats_text += f'Error Std: {error_std:.4f}'

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure
        # Replace special characters in filename
        safe_filename = col.replace('/', '_').replace('\\', '_').replace(' ', '_')
        fig_path = plots_dir / f'{safe_filename}.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {fig_path.name}")

    # Create a summary plot with all data (optional)
    print("\nCreating summary plots by category...")

    # Group similar columns
    column_groups = {
        'dof_torque': [col for col in columns_to_plot if 'dof_torque' in col],
        'raw_action': [col for col in columns_to_plot if 'raw_action' in col],
        'processed_action': [col for col in columns_to_plot if 'processed_action' in col],
        'target_fingertip_pos': [col for col in columns_to_plot if 'target_fingertip_pos' in col],
        'fixed_pos_obs_frame': [col for col in columns_to_plot if 'fixed_pos_obs_frame' in col],
        'fingertip_midpoint_pos': [col for col in columns_to_plot if 'fingertip_midpoint_pos' in col],
        'current_joint_pos': [col for col in columns_to_plot if 'current_joint_pos' in col],
        'applied_dof_torque': [col for col in columns_to_plot if 'applied_dof_torque' in col],
        'target_joint_pos': [col for col in columns_to_plot if 'target_joint_pos' in col],
    }

    # Create grouped plots
    for group_name, cols in column_groups.items():
        if not cols:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))

        for col in cols:
            y_data = df[col].values
            ax.plot(x_data, y_data, linewidth=1.5, alpha=0.7, label=f'{col} (Actual)', marker='o', markersize=3)

            # Add desired position if available for fingertip positions
            if col in desired_positions:
                desired_y = desired_positions[col]
                ax.plot(x_data, desired_y, linewidth=1.5, alpha=0.7, linestyle='--',
                       label=f'{col} (Desired)', marker='x', markersize=3)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        title = f'{group_name} - All Components'
        if group_name == 'fingertip_midpoint_pos' and desired_positions:
            title += ' (Actual vs Desired)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        plt.tight_layout()

        # Save grouped figure
        fig_path = plots_dir / f'grouped_{group_name}.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved grouped plot: {fig_path.name}")

    # Create a special comparison plot for positions (X, Y, Z in separate subplots)
    if desired_positions:
        print("\nCreating position tracking comparison plot...")
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        position_axes = ['X', 'Y', 'Z']
        position_cols = ['fingertip_midpoint_pos_0', 'fingertip_midpoint_pos_1', 'fingertip_midpoint_pos_2']
        colors = ['red', 'green', 'blue']

        for i, (ax, col, axis_name, color) in enumerate(zip(axes, position_cols, position_axes, colors)):
            actual_data = df[col].values
            desired_data = desired_positions[col]
            error = actual_data - desired_data

            # Plot actual and desired
            ax.plot(x_data, actual_data, color=color, linewidth=2, alpha=0.8,
                   label=f'Actual {axis_name}', marker='o', markersize=4)
            ax.plot(x_data, desired_data, color=color, linewidth=2, alpha=0.6,
                   linestyle='--', label=f'Desired {axis_name}', marker='x', markersize=4)

            # Add error statistics
            error_mean = np.mean(np.abs(error))
            error_std = np.std(error)
            error_max = np.max(np.abs(error))

            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(f'Position {axis_name} (m)', fontsize=11)
            ax.set_title(f'{axis_name}-Direction Position Tracking', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)

            # Add error text box
            error_text = f'Mean |Error|: {error_mean:.4f}\n'
            error_text += f'Max |Error|: {error_max:.4f}\n'
            error_text += f'Error Std: {error_std:.4f}'

            ax.text(0.98, 0.98, error_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        fig.suptitle('End-Effector Position Tracking (Actual vs Desired)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save comparison figure
        fig_path = plots_dir / 'position_tracking_comparison.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved position tracking comparison: {fig_path.name}")

    print(f"\nAll plots saved successfully!")
    print(f"Total individual plots: {len(columns_to_plot)}")
    print(f"Total grouped plots: {len([g for g in column_groups.values() if g])}")
    if desired_positions:
        print(f"Position tracking plots: 4 (3 individual + 1 combined comparison)")


if __name__ == "__main__":
    # Paths
    csv_path = '/home/tih_auto_hpz4/Aditya/IsaacLab/scripts/peg_hole_2/logs/ppo_factory/csv_files/diff_ik_low_T_high_bounds.csv'
    output_dir = '/home/tih_auto_hpz4/Aditya/IsaacLab/scripts/peg_hole_2/logs/ppo_factory'

    # Generate plots
    plot_csv_data(csv_path, output_dir)
