#!/usr/bin/env python3
"""
Test script for IsaacLab Factory Environment (factory_env_diff_ik.py)
Applies a constant action and plots the results similar to sample_test_real.py
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

# Import IsaacLab components
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test IsaacLab peg-hole environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--n_steps", type=int, default=24*8, help="Number of RL steps to run.")
parser.add_argument("--x_action", type=float, default=0.0, help="X-direction action magnitude (-1.0 to 1.0).")
parser.add_argument("--y_action", type=float, default=0.1, help="Y-direction action magnitude (-1.0 to 1.0).")
# parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI).")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
from isaaclab_tasks.utils import parse_env_cfg

# Add Isaac Lab root directory to Python path
ISAACLAB_ROOT = Path(__file__).resolve().parents[2]
if str(ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_ROOT))


def get_next_run_number(test_runs_dir):
    """Get the next run number by checking existing folders."""
    if not test_runs_dir.exists():
        return 1

    existing_runs = [d.name for d in test_runs_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not existing_runs:
        return 1

    run_numbers = []
    for run_name in existing_runs:
        try:
            num = int(run_name.split('_')[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue

    return max(run_numbers) + 1 if run_numbers else 1


def main():
    # Configuration
    N_STEPS = args_cli.n_steps
    X_ACTION = args_cli.x_action
    Y_ACTION = args_cli.y_action

    # Setup output directory
    test_runs_dir = Path(__file__).parent / 'test_runs'
    run_number = get_next_run_number(test_runs_dir)
    run_dir = test_runs_dir / f'run_{run_number:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Test] Run number: {run_number}")
    print(f"[Test] Saving results to: {run_dir}")

    # Register and create environment
    env_id = "peg_insert-v0-test"
    gym.register(
        id=env_id,
        entry_point="scripts.peg_hole_2.absolute_target_env:FactoryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "scripts.peg_hole_2.factory_env_cfg_diff_ik:FactoryTaskPegInsertCfg",
        },
    )

    env_cfg = parse_env_cfg(env_id, num_envs=args_cli.num_envs)
    env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.enable_env_tune_changes(True)

    # Get environment parameters
    num_envs = env.unwrapped.scene.num_envs
    device = env.unwrapped.device
    physics_dt = env.unwrapped.physics_dt
    decimation = env.unwrapped.cfg.decimation

    print(f"[Test] Number of environments: {num_envs}")
    print(f"[Test] Physics dt: {physics_dt:.6f} sec")
    print(f"[Test] Decimation: {decimation}")
    print(f"[Test] Control rate: {1.0 / (physics_dt * decimation):.2f} Hz")

    # Reset environment
    print("[Test] Resetting environment...")
    obs, _ = env.reset()

    # Data storage for first environment only
    log_data = {
        'timestamps': [],
        'eeftip_pos': [],
        'eeftip_linvel': [],
        'eeftip_angvel': [],
        'actions': [],
    }

    # Create action: x-direction only, all other components zero
    action = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)
    action[:, 0] = X_ACTION
    action[:, 1] = Y_ACTION

    print(f"[Test] Running {N_STEPS} steps with initial action: {action[0].cpu().numpy()}")
    print(f"[Test] Will collect {N_STEPS} RL-level samples")

    start_time = time.time()
    start_timestamp = time.time()

    # Run N steps with actions
    for step in range(N_STEPS):
        # Alternate direction every 6 steps (similar to real env test)
        # if step % 6 == 0:
        #     action = -action

        print(f"\n[Test] Step {step + 1}/{N_STEPS}, Action: {action[0].cpu().numpy()}")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Get logged data from environment (for first env only)
        env_unwrapped = env.unwrapped

        # Extract end-effector data
        eeftip_pos = env_unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
        eeftip_linvel = env_unwrapped.ee_linvel_fd[0].cpu().numpy()
        eeftip_angvel = env_unwrapped.ee_angvel_fd[0].cpu().numpy()

        # Store data
        log_data['timestamps'].append(time.time() - start_timestamp)
        log_data['eeftip_pos'].append(eeftip_pos)
        log_data['eeftip_linvel'].append(eeftip_linvel)
        log_data['eeftip_angvel'].append(eeftip_angvel)
        log_data['actions'].append(action[0].cpu().numpy())

        # Print current state
        print(f"  EE Pos: {eeftip_pos}")
        print(f"  EE LinVel: {eeftip_linvel}")
        print(f"  EE AngVel: {eeftip_angvel}")

    print(f"\n[Test] Completed {N_STEPS} steps in {time.time() - start_time:.2f} seconds")
    print(f"[Test] Collected {len(log_data['timestamps'])} samples")

    # Process logged data
    if len(log_data['timestamps']) == 0:
        print("[Test] No data collected! Exiting.")
        env.close()
        simulation_app.close()
        return

    # Convert to numpy arrays
    rl_steps = np.arange(len(log_data['timestamps']))
    timestamps = np.array(log_data['timestamps'])
    eeftip_pos_x = np.array([pos[0] for pos in log_data['eeftip_pos']])
    eeftip_pos_y = np.array([pos[1] for pos in log_data['eeftip_pos']])
    eeftip_pos_z = np.array([pos[2] for pos in log_data['eeftip_pos']])
    eeftip_linvel_x = np.array([vel[0] for vel in log_data['eeftip_linvel']])
    eeftip_linvel_y = np.array([vel[1] for vel in log_data['eeftip_linvel']])
    eeftip_linvel_z = np.array([vel[2] for vel in log_data['eeftip_linvel']])
    eeftip_angvel_x = np.array([vel[0] for vel in log_data['eeftip_angvel']])
    eeftip_angvel_y = np.array([vel[1] for vel in log_data['eeftip_angvel']])
    eeftip_angvel_z = np.array([vel[2] for vel in log_data['eeftip_angvel']])

    # Calculate desired positions by accumulating actions
    actions_array = np.array(log_data['actions'])
    initial_pos_x = eeftip_pos_x[0]
    initial_pos_y = eeftip_pos_y[0]
    initial_pos_z = eeftip_pos_z[0]

    # Cumulative sum of actions (only position components: 0, 1, 2)
    # desired_pos_x = initial_pos_x + np.cumsum(actions_array[:, 0])
    # desired_pos_y = initial_pos_y + np.cumsum(actions_array[:, 1])
    # desired_pos_z = initial_pos_z + np.cumsum(actions_array[:, 2])
    desired_pos_x = initial_pos_x + actions_array[:, 0]
    desired_pos_y = initial_pos_y + actions_array[:, 1]
    desired_pos_z = initial_pos_z + actions_array[:, 2]

    # Calculate tracking errors
    error_x = eeftip_pos_x - desired_pos_x
    error_y = eeftip_pos_y - desired_pos_y
    error_z = eeftip_pos_z - desired_pos_z

    print(f"[Test] Position tracking errors:")
    print(f"  X: Mean |Error| = {np.mean(np.abs(error_x)):.6f} m, Std = {np.std(error_x):.6f} m")
    print(f"  Y: Mean |Error| = {np.mean(np.abs(error_y)):.6f} m, Std = {np.std(error_y):.6f} m")
    print(f"  Z: Mean |Error| = {np.mean(np.abs(error_z)):.6f} m, Std = {np.std(error_z):.6f} m")

    # Create plots
    print("[Test] Creating plots...")

    # ========== Original Combined Plots ==========
    # Figure 1: RL Step vs End-Effector States (Combined)
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10))
    fig1.suptitle(f'RL Step vs End-Effector States (X Action = {X_ACTION}, {N_STEPS} RL steps)', fontsize=16)

    # Plot 1a: EE Position
    axes1[0].plot(rl_steps, eeftip_pos_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
    axes1[0].plot(rl_steps, eeftip_pos_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
    axes1[0].plot(rl_steps, eeftip_pos_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
    # axes1[0].plot(rl_steps, eeftip_pos_x, 'ro', markersize=6, label='RL Step', zorder=5)
    axes1[0].set_xlabel('RL Step')
    axes1[0].set_ylabel('Position (m)')
    axes1[0].set_title('End-Effector Position')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)

    # Plot 1b: EE Linear Velocity
    axes1[1].plot(rl_steps, eeftip_linvel_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
    axes1[1].plot(rl_steps, eeftip_linvel_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
    axes1[1].plot(rl_steps, eeftip_linvel_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
    # axes1[1].plot(rl_steps, eeftip_linvel_x, 'ko', markersize=6, label='RL Step', zorder=5)
    axes1[1].set_xlabel('RL Step')
    axes1[1].set_ylabel('Linear Velocity (m/s)')
    axes1[1].set_title('End-Effector Linear Velocity')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)

    # Plot 1c: EE Angular Velocity
    axes1[2].plot(rl_steps, eeftip_angvel_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
    axes1[2].plot(rl_steps, eeftip_angvel_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
    axes1[2].plot(rl_steps, eeftip_angvel_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
    # axes1[2].plot(rl_steps, eeftip_angvel_x, 'ko', markersize=6, label='RL Step', zorder=5)
    axes1[2].set_xlabel('RL Step')
    axes1[2].set_ylabel('Angular Velocity (rad/s)')
    axes1[2].set_title('End-Effector Angular Velocity')
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save Figure 1
    fig1_path = run_dir / 'rl_step_vs_states.png'
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"[Test] Saved: {fig1_path}")
    plt.close(fig1)

    # Figure 2: Time vs End-Effector States (Combined)
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
    fig2.suptitle(f'Time vs End-Effector States (X Action = {X_ACTION}, {N_STEPS} RL steps)', fontsize=16)

    # Plot 2a: EE Position
    axes2[0].plot(timestamps, eeftip_pos_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
    axes2[0].plot(timestamps, eeftip_pos_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
    axes2[0].plot(timestamps, eeftip_pos_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
    # axes2[0].plot(timestamps, eeftip_pos_x, 'ro', markersize=6, label='RL Step', zorder=5)
    axes2[0].set_xlabel('Time (s)')
    axes2[0].set_ylabel('Position (m)')
    axes2[0].set_title('End-Effector Position')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    # Plot 2b: EE Linear Velocity
    axes2[1].plot(timestamps, eeftip_linvel_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
    axes2[1].plot(timestamps, eeftip_linvel_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
    axes2[1].plot(timestamps, eeftip_linvel_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
    # axes2[1].plot(timestamps, eeftip_linvel_x, 'ko', markersize=6, label='RL Step', zorder=5)
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_ylabel('Linear Velocity (m/s)')
    axes2[1].set_title('End-Effector Linear Velocity')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    # Plot 2c: EE Angular Velocity
    axes2[2].plot(timestamps, eeftip_angvel_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
    axes2[2].plot(timestamps, eeftip_angvel_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
    axes2[2].plot(timestamps, eeftip_angvel_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
    # axes2[2].plot(timestamps, eeftip_angvel_x, 'ko', markersize=6, label='RL Step', zorder=5)
    axes2[2].set_xlabel('Time (s)')
    axes2[2].set_ylabel('Angular Velocity (rad/s)')
    axes2[2].set_title('End-Effector Angular Velocity')
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save Figure 2
    fig2_path = run_dir / 'time_vs_states.png'
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"[Test] Saved: {fig2_path}")
    plt.close(fig2)

    # ========== Individual Direction Plots ==========
    # Create separate figures for each direction and time series
    # Data for plotting
    plot_data = {
        'rl_steps': {
            'x_axis': rl_steps,
            'x_label': 'RL Step',
            'prefix': 'rl_step'
        },
        'time': {
            'x_axis': timestamps,
            'x_label': 'Time (s)',
            'prefix': 'time'
        }
    }

    # Position, velocity, and angular velocity data
    ee_data = {
        'position': {
            'x': eeftip_pos_x,
            'y': eeftip_pos_y,
            'z': eeftip_pos_z,
            'y_label': 'Position (m)',
            'title': 'End-Effector Position'
        },
        'linear_velocity': {
            'x': eeftip_linvel_x,
            'y': eeftip_linvel_y,
            'z': eeftip_linvel_z,
            'y_label': 'Linear Velocity (m/s)',
            'title': 'End-Effector Linear Velocity'
        },
        'angular_velocity': {
            'x': eeftip_angvel_x,
            'y': eeftip_angvel_y,
            'z': eeftip_angvel_z,
            'y_label': 'Angular Velocity (rad/s)',
            'title': 'End-Effector Angular Velocity'
        }
    }

    # Create individual plots for each combination
    for x_type, x_info in plot_data.items():
        for data_type, data_info in ee_data.items():
            # Create figure with 3 subplots (one for each direction)
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # Check if this is position data (add desired positions)
            is_position = data_type == 'position'

            title = f'{data_info["title"]} vs {x_info["x_label"]} (X Action = {X_ACTION}, {N_STEPS} RL steps)'
            if is_position:
                title += '\n(Actual vs Desired)'
            fig.suptitle(title, fontsize=16)

            # Plot X direction
            axes[0].plot(x_info['x_axis'], data_info['x'], 'r-', linewidth=2, alpha=0.8, label='Actual X')
            # axes[0].plot(x_info['x_axis'], data_info['x'], 'ro', markersize=6, zorder=5)
            if is_position:
                axes[0].plot(x_info['x_axis'], desired_pos_x, 'r--', linewidth=2, alpha=0.6, label='Desired X')
                # axes[0].plot(x_info['x_axis'], desired_pos_x, 'rx', markersize=6, zorder=5)
            axes[0].set_xlabel(x_info['x_label'])
            axes[0].set_ylabel(f'{data_info["y_label"]} (X)')
            axes[0].set_title('X Direction')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot Y direction
            axes[1].plot(x_info['x_axis'], data_info['y'], 'g-', linewidth=2, alpha=0.8, label='Actual Y')
            # axes[1].plot(x_info['x_axis'], data_info['y'], 'go', markersize=6, zorder=5)
            if is_position:
                axes[1].plot(x_info['x_axis'], desired_pos_y, 'g--', linewidth=2, alpha=0.6, label='Desired Y')
                # axes[1].plot(x_info['x_axis'], desired_pos_y, 'gx', markersize=6, zorder=5)
            axes[1].set_xlabel(x_info['x_label'])
            axes[1].set_ylabel(f'{data_info["y_label"]} (Y)')
            axes[1].set_title('Y Direction')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Plot Z direction
            axes[2].plot(x_info['x_axis'], data_info['z'], 'b-', linewidth=2, alpha=0.8, label='Actual Z')
            # axes[2].plot(x_info['x_axis'], data_info['z'], 'bo', markersize=6, zorder=5)
            if is_position:
                axes[2].plot(x_info['x_axis'], desired_pos_z, 'b--', linewidth=2, alpha=0.6, label='Desired Z')
                # axes[2].plot(x_info['x_axis'], desired_pos_z, 'bx', markersize=6, zorder=5)
            axes[2].set_xlabel(x_info['x_label'])
            axes[2].set_ylabel(f'{data_info["y_label"]} (Z)')
            axes[2].set_title('Z Direction')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            fig_path = run_dir / f'{x_info["prefix"]}_vs_{data_type}_individual.png'
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"[Test] Saved: {fig_path}")
            plt.close(fig)

    # Save data to numpy file
    data_path = run_dir / 'data.npz'
    np.savez(data_path,
             rl_steps=rl_steps,
             timestamps=timestamps,
             eeftip_pos_x=eeftip_pos_x,
             eeftip_pos_y=eeftip_pos_y,
             eeftip_pos_z=eeftip_pos_z,
             desired_pos_x=desired_pos_x,
             desired_pos_y=desired_pos_y,
             desired_pos_z=desired_pos_z,
             error_x=error_x,
             error_y=error_y,
             error_z=error_z,
             eeftip_linvel_x=eeftip_linvel_x,
             eeftip_linvel_y=eeftip_linvel_y,
             eeftip_linvel_z=eeftip_linvel_z,
             eeftip_angvel_x=eeftip_angvel_x,
             eeftip_angvel_y=eeftip_angvel_y,
             eeftip_angvel_z=eeftip_angvel_z,
             actions=actions_array,
             n_steps=N_STEPS,
             x_action=X_ACTION,
             decimation=decimation,
             physics_dt=physics_dt)
    print(f"[Test] Saved data: {data_path}")

    # Save test configuration
    config_path = run_dir / 'config.txt'
    with open(config_path, 'w') as f:
        f.write(f"Test Run Configuration\n")
        f.write(f"=====================\n")
        f.write(f"Run Number: {run_number}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nParameters:\n")
        f.write(f"  N_STEPS: {N_STEPS}\n")
        f.write(f"  X_ACTION: {X_ACTION}\n")
        f.write(f"  Decimation: {decimation}\n")
        f.write(f"  Physics dt: {physics_dt:.6f} sec\n")
        f.write(f"  Control rate: {1.0 / (physics_dt * decimation):.2f} Hz\n")
        f.write(f"  Number of environments: {num_envs}\n")
        f.write(f"\nData Collected:\n")
        f.write(f"  Total RL samples: {len(log_data['timestamps'])}\n")
        f.write(f"  Duration: {timestamps[-1]:.3f} seconds\n")
        f.write(f"\nPosition Tracking Performance:\n")
        f.write(f"  X-axis:\n")
        f.write(f"    Mean |Error|: {np.mean(np.abs(error_x)):.6f} m\n")
        f.write(f"    Max |Error|:  {np.max(np.abs(error_x)):.6f} m\n")
        f.write(f"    Std Error:    {np.std(error_x):.6f} m\n")
        f.write(f"  Y-axis:\n")
        f.write(f"    Mean |Error|: {np.mean(np.abs(error_y)):.6f} m\n")
        f.write(f"    Max |Error|:  {np.max(np.abs(error_y)):.6f} m\n")
        f.write(f"    Std Error:    {np.std(error_y):.6f} m\n")
        f.write(f"  Z-axis:\n")
        f.write(f"    Mean |Error|: {np.mean(np.abs(error_z)):.6f} m\n")
        f.write(f"    Max |Error|:  {np.max(np.abs(error_z)):.6f} m\n")
        f.write(f"    Std Error:    {np.std(error_z):.6f} m\n")
    print(f"[Test] Saved config: {config_path}")

    # Cleanup
    env.close()
    print("[Test] Test completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Test] Interrupted by user")
    except Exception as e:
        print(f"\n[Test] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
