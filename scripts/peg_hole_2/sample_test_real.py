#!/usr/bin/env python3
"""
Test script for env_peg_hole_multiprocess.py
Applies a constant x-direction action and plots the results.
"""

import sys
import time
import signal
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import the environment
from env_peg_hole_multiprocess import EnvPegHole, joint_state_subscriber_process


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
    N_STEPS = 15  # Number of RL steps to run
    X_ACTION = 1  # X-direction action magnitude (-1.0 to 1.0)

    # Setup output directory
    test_runs_dir = Path(__file__).parent / 'test_runs'
    run_number = get_next_run_number(test_runs_dir)
    run_dir = test_runs_dir / f'run_{run_number:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Test] Run number: {run_number}")
    print(f"[Test] Saving results to: {run_dir}")

    joint_names = [
        "joint_1", "joint_2", "joint_3",
        "joint_4", "joint_5", "joint_6", "joint_7",
    ]
    n_joints = len(joint_names)

    # Create shared memory for joint states
    shm = shared_memory.SharedMemory(create=True, size=n_joints * 8)
    shm_array = np.ndarray((n_joints,), dtype=np.float64, buffer=shm.buf)
    shm_array[:] = 0.0

    # Start subscriber process
    sub_proc = mp.Process(
        target=joint_state_subscriber_process,
        args=(shm.name, n_joints, joint_names),
        daemon=True,
    )
    sub_proc.start()
    print(f"[Test] Subscriber process started (PID={sub_proc.pid})")

    # Initialize environment
    env = EnvPegHole(shm.name, n_joints, joint_names)
    ctrl_period = 1.0 / env.ctrl_rate

    # Cleanup handler
    def cleanup(_sig=None, _frame=None):
        print("\n[Test] Cleaning up...")
        sub_proc.terminate()
        sub_proc.join(timeout=2.0)
        shm.close()
        shm.unlink()
        env.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Reset environment
        print("[Test] Resetting environment...")
        env.reset()

        # Create action: x-direction only, all other components zero
        action = np.array([X_ACTION, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(f"[Test] Running {N_STEPS} steps with action: {action}")
        print(f"[Test] Decimation: {env.decimation} (will collect {N_STEPS * env.decimation} decimation-level samples)")

        # Enable logging at decimation level
        env.enable_logging()

        start_time = time.time()

        # Run N steps with the same action
        for step in range(N_STEPS):
            if step % 2 == 0:
                X_ACTION = -X_ACTION
            action = np.array([X_ACTION, 0.0, 0.0, 0.0, 0.0, 0.0])
            print(f"\n[Test] Step {step + 1}/{N_STEPS}")
            env.step(action)
            time.sleep(ctrl_period)

        print(f"\n[Test] Completed {N_STEPS} steps in {time.time() - start_time:.2f} seconds")

        # Get decimation-level logged data
        decimation_log = env.get_decimation_log()
        print(f"[Test] Collected {len(decimation_log)} decimation-level samples")

        # Process logged data
        if len(decimation_log) == 0:
            print("[Test] No data collected! Exiting.")
            cleanup()
            return

        # Extract data from log
        decimation_steps = []
        timestamps = []
        eeftip_pos_x = []
        eeftip_pos_y = []
        eeftip_pos_z = []
        eeftip_linvel_x = []
        eeftip_linvel_y = []
        eeftip_linvel_z = []
        eeftip_angvel_x = []
        eeftip_angvel_y = []
        eeftip_angvel_z = []

        # Normalize timestamps to start from 0
        start_timestamp = decimation_log[0]['timestamp']

        for i, entry in enumerate(decimation_log):
            decimation_steps.append(i)
            timestamps.append(entry['timestamp'] - start_timestamp)
            eeftip_pos_x.append(entry['eeftip_pos'][0])
            eeftip_pos_y.append(entry['eeftip_pos'][1])
            eeftip_pos_z.append(entry['eeftip_pos'][2])
            eeftip_linvel_x.append(entry['eeftip_linvel'][0])
            eeftip_linvel_y.append(entry['eeftip_linvel'][1])
            eeftip_linvel_z.append(entry['eeftip_linvel'][2])
            eeftip_angvel_x.append(entry['eeftip_angvel'][0])
            eeftip_angvel_y.append(entry['eeftip_angvel'][1])
            eeftip_angvel_z.append(entry['eeftip_angvel'][2])

        # Identify RL step boundaries (every decimation samples)
        # The first sample of each RL step will be marked
        rl_step_indices = list(range(0, len(decimation_log), env.decimation))
        rl_step_decimation = [decimation_steps[i] for i in rl_step_indices if i < len(decimation_steps)]
        rl_step_timestamps = [timestamps[i] for i in rl_step_indices if i < len(timestamps)]

        # Convert to numpy arrays
        decimation_steps = np.array(decimation_steps)
        timestamps = np.array(timestamps)
        eeftip_pos_x = np.array(eeftip_pos_x)
        eeftip_pos_y = np.array(eeftip_pos_y)
        eeftip_pos_z = np.array(eeftip_pos_z)
        eeftip_linvel_x = np.array(eeftip_linvel_x)
        eeftip_linvel_y = np.array(eeftip_linvel_y)
        eeftip_linvel_z = np.array(eeftip_linvel_z)
        eeftip_angvel_x = np.array(eeftip_angvel_x)
        eeftip_angvel_y = np.array(eeftip_angvel_y)
        eeftip_angvel_z = np.array(eeftip_angvel_z)

        # Create plots
        print("[Test] Creating plots...")

        # Figure 1: Decimation Step vs EE Position, Linear Velocity, Angular Velocity
        fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10))
        fig1.suptitle(f'Decimation Step vs End-Effector States (X Action = {X_ACTION}, {N_STEPS} RL steps)', fontsize=16)

        # Plot 1a: EE Position
        axes1[0].plot(decimation_steps, eeftip_pos_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
        axes1[0].plot(decimation_steps, eeftip_pos_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
        axes1[0].plot(decimation_steps, eeftip_pos_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
        # Add markers at RL step boundaries
        axes1[0].plot(rl_step_decimation, [eeftip_pos_x[i] for i in rl_step_indices if i < len(eeftip_pos_x)],
                     'ro', markersize=6, label='RL Step Start', zorder=5)
        axes1[0].set_xlabel('Decimation Step')
        axes1[0].set_ylabel('Position (m)')
        axes1[0].set_title('End-Effector Position')
        axes1[0].legend()
        axes1[0].grid(True, alpha=0.3)

        # Plot 1b: EE Linear Velocity
        axes1[1].plot(decimation_steps, eeftip_linvel_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
        axes1[1].plot(decimation_steps, eeftip_linvel_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
        axes1[1].plot(decimation_steps, eeftip_linvel_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
        # Add markers at RL step boundaries
        axes1[1].plot(rl_step_decimation, [eeftip_linvel_x[i] for i in rl_step_indices if i < len(eeftip_linvel_x)],
                     'ko', markersize=6, label='RL Step Start', zorder=5)
        axes1[1].set_xlabel('Decimation Step')
        axes1[1].set_ylabel('Linear Velocity (m/s)')
        axes1[1].set_title('End-Effector Linear Velocity')
        axes1[1].legend()
        axes1[1].grid(True, alpha=0.3)

        # Plot 1c: EE Angular Velocity
        axes1[2].plot(decimation_steps, eeftip_angvel_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
        axes1[2].plot(decimation_steps, eeftip_angvel_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
        axes1[2].plot(decimation_steps, eeftip_angvel_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
        # Add markers at RL step boundaries
        axes1[2].plot(rl_step_decimation, [eeftip_angvel_x[i] for i in rl_step_indices if i < len(eeftip_angvel_x)],
                     'ko', markersize=6, label='RL Step Start', zorder=5)
        axes1[2].set_xlabel('Decimation Step')
        axes1[2].set_ylabel('Angular Velocity (rad/s)')
        axes1[2].set_title('End-Effector Angular Velocity')
        axes1[2].legend()
        axes1[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save Figure 1
        fig1_path = run_dir / 'decimation_step_vs_states.png'
        fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
        print(f"[Test] Saved: {fig1_path}")

        # Figure 2: Time vs EE Position, Linear Velocity, Angular Velocity
        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
        fig2.suptitle(f'Time vs End-Effector States (X Action = {X_ACTION}, {N_STEPS} RL steps)', fontsize=16)

        # Plot 2a: EE Position
        axes2[0].plot(timestamps, eeftip_pos_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
        axes2[0].plot(timestamps, eeftip_pos_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
        axes2[0].plot(timestamps, eeftip_pos_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
        # Add markers at RL step boundaries
        axes2[0].plot(rl_step_timestamps, [eeftip_pos_x[i] for i in rl_step_indices if i < len(eeftip_pos_x)],
                     'ro', markersize=6, label='RL Step Start', zorder=5)
        axes2[0].set_xlabel('Time (s)')
        axes2[0].set_ylabel('Position (m)')
        axes2[0].set_title('End-Effector Position')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        # Plot 2b: EE Linear Velocity
        axes2[1].plot(timestamps, eeftip_linvel_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
        axes2[1].plot(timestamps, eeftip_linvel_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
        axes2[1].plot(timestamps, eeftip_linvel_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
        # Add markers at RL step boundaries
        axes2[1].plot(rl_step_timestamps, [eeftip_linvel_x[i] for i in rl_step_indices if i < len(eeftip_linvel_x)],
                     'ko', markersize=6, label='RL Step Start', zorder=5)
        axes2[1].set_xlabel('Time (s)')
        axes2[1].set_ylabel('Linear Velocity (m/s)')
        axes2[1].set_title('End-Effector Linear Velocity')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)

        # Plot 2c: EE Angular Velocity
        axes2[2].plot(timestamps, eeftip_angvel_x, 'r-', label='X', linewidth=1.5, alpha=0.7)
        axes2[2].plot(timestamps, eeftip_angvel_y, 'g-', label='Y', linewidth=1.5, alpha=0.7)
        axes2[2].plot(timestamps, eeftip_angvel_z, 'b-', label='Z', linewidth=1.5, alpha=0.7)
        # Add markers at RL step boundaries
        axes2[2].plot(rl_step_timestamps, [eeftip_angvel_x[i] for i in rl_step_indices if i < len(eeftip_angvel_x)],
                     'ko', markersize=6, label='RL Step Start', zorder=5)
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

        # Save data to numpy file
        data_path = run_dir / 'data.npz'
        np.savez(data_path,
                 decimation_steps=decimation_steps,
                 timestamps=timestamps,
                 eeftip_pos_x=eeftip_pos_x,
                 eeftip_pos_y=eeftip_pos_y,
                 eeftip_pos_z=eeftip_pos_z,
                 eeftip_linvel_x=eeftip_linvel_x,
                 eeftip_linvel_y=eeftip_linvel_y,
                 eeftip_linvel_z=eeftip_linvel_z,
                 eeftip_angvel_x=eeftip_angvel_x,
                 eeftip_angvel_y=eeftip_angvel_y,
                 eeftip_angvel_z=eeftip_angvel_z,
                 n_steps=N_STEPS,
                 x_action=X_ACTION,
                 decimation=env.decimation)
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
            f.write(f"  Decimation: {env.decimation}\n")
            f.write(f"  Control Rate: {env.ctrl_rate} Hz\n")
            f.write(f"  Physics dt: {env.physics_dt:.6f} sec\n")
            f.write(f"\nData Collected:\n")
            f.write(f"  Total decimation samples: {len(decimation_log)}\n")
            f.write(f"  Duration: {timestamps[-1]:.3f} seconds\n")
        print(f"[Test] Saved config: {config_path}")

        # Show all plots
        print("[Test] Displaying plots. Close the plot windows to exit.")
        plt.show()

    except KeyboardInterrupt:
        print("\n[Test] Interrupted by user")
    except Exception as e:
        print(f"\n[Test] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
