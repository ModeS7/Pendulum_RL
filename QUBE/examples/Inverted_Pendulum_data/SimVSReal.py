import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import numba as nb
from numpy.ma.core import arctan2

# System Parameters from the system identification document (Table 3)
Rm = 8.94  # Motor resistance (Ohm)
Km = 0.042  # Motor back-emf constant
Jm = 6e-5  # Total moment of inertia acting on motor shaft (kg·m^2)
bm = 3e-4  # Viscous damping coefficient (Nm/rad/s)
DA = 3e-4  # Damping coefficient of pendulum arm (Nm/rad/s)
DL = 5e-4  # Damping coefficient of pendulum link (Nm/rad/s)
mA = 0.095  # Weight of pendulum arm (kg)
mL = 0.024  # Weight of pendulum link (kg)
LA = 0.085  # Length of pendulum arm (m)
LL = 0.129  # Length of pendulum link (m)
#JA = 5.72e-5  # Inertia moment of pendulum arm (kg·m^2)
#JL = 1.31e-4  # Inertia moment of pendulum link (kg·m^2)
g = 9.81  # Gravity constant (m/s^2)
JA = mA * LA ** 2 * 7 / 48  # Pendulum arm moment of inertia (kg·m²)
JL = mL * LL ** 2 / 3  # Pendulum link moment of inertia (kg·m²)
l_1 = LL / 2
k = 0.002
# Pre-compute constants for optimization
half_mL_LL_g = 0.5 * mL * LL * g
half_mL_LL_LA = 0.5 * mL * LL * LA
quarter_mL_LL_squared = 0.25 * mL * LL ** 2
Mp_g_Lp = mL * g * LL
Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia (kg·m²)

voltage_deadzone = 0.2  # Dead zone for motor voltage
max_voltage = 10.0  # Maximum motor voltage
THETA_MIN = -2.0  # Minimum arm angle (radians)
THETA_MAX = 2.0  # Maximum arm angle (radians)


# -------------------- CUSTOM MATH OPERATIONS --------------------
@nb.njit(fastmath=True, cache=True)
def clip_value(value, min_value, max_value):
    """Fast custom clip function"""
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value


@nb.njit(fastmath=True, cache=True)
def apply_voltage_deadzone(vm):
    """Apply motor voltage dead zone"""
    if -voltage_deadzone <= vm <= voltage_deadzone:
        vm = 0.0
    return vm


@nb.njit(fastmath=True, cache=True)
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


# -------------------- SIMPLIFIED INTEGRATOR --------------------
@nb.njit(fastmath=True, cache=True)
def enforce_theta_limits(state):
    """Enforce hard limits on theta angle and velocity"""
    theta, alpha, theta_dot, alpha_dot = state

    # Apply hard limit on theta
    if theta > THETA_MAX:
        theta = THETA_MAX
        # If hitting upper limit with positive velocity, stop the motion
        if theta_dot > 0:
            theta_dot = 0.0
    elif theta < THETA_MIN:
        theta = THETA_MIN
        # If hitting lower limit with negative velocity, stop the motion
        if theta_dot < 0:
            theta_dot = 0.0

    return np.array([theta, alpha, theta_dot, alpha_dot])


@nb.njit(fastmath=True, cache=True)
def dynamics_step(state, t, vm):
    """Updated dynamics calculation to match the new simulation model"""
    theta_0, theta_1, theta_0_dot, theta_1_dot = state

    # Calculate sines and cosines for dynamics
    s0, c0 = np.sin(theta_0), np.cos(theta_0)
    s1, c1 = np.sin(theta_1), np.cos(theta_1)

    # Check theta limits - implement hard stops
    if (theta_0 >= THETA_MAX and theta_0_dot > 0) or (theta_0 <= THETA_MIN and theta_0_dot < 0):
        theta_0_dot = 0.0  # Stop the arm motion at the limits

    # Apply dead zone and calculate motor torque
    vm = apply_voltage_deadzone(vm)

    # Motor torque calculation
    torque = Km * (vm - Km * theta_0_dot) / Rm

    # Set up the mass matrix and force vector according to the new model
    alpha = JA + mL * LA ** 2 + mL * l_1 ** 2 * s1 ** 2
    beta = -mL * l_1 ** 2 * (2 * s1 * c1)
    gamma = -mL * LA * l_1 * c1
    sigma = mL * LA * l_1 * s1

    # Set up the mass matrix (with sign adjustments to match file 2)
    M = np.array([
        [-alpha, -gamma],
        [-gamma, -(JL + mL * l_1 ** 2)]
    ])

    # Right-hand side force vector (matching file 2)
    f = np.array([
        -torque + DA * theta_0_dot + k * arctan2(s0, c0) + sigma * theta_1_dot ** 2 - beta * theta_0_dot * theta_1_dot,
        DL * theta_1_dot + mL * g * l_1 * s1 + 0.5 * beta * theta_0_dot ** 2
    ])

    # Solve for accelerations using numpy's solve
    # For numba compatibility, we'll solve manually
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

    # Handle near-singular matrix
    if abs(det_M) < 1e-10:
        theta_0_ddot = 0
        theta_1_ddot = 0
    else:
        # Solve for accelerations (Cramer's rule)
        theta_0_ddot = (M[1, 1] * f[0] - M[0, 1] * f[1]) / det_M
        theta_1_ddot = (M[0, 0] * f[1] - M[1, 0] * f[0]) / det_M

    return np.array([theta_0_dot, theta_1_dot, theta_0_ddot, theta_1_ddot])


def rk4_integration(state0, t_span, dt, voltage_data, time_data):
    """
    4th-order Runge-Kutta integrator using real voltage data
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    # Pre-allocate arrays for results
    t = np.linspace(t_start, t_end, n_steps)
    states = np.zeros((4, n_steps))

    # Set initial state with limits applied
    states[:, 0] = enforce_theta_limits(state0)

    # Integration loop
    for i in range(1, n_steps):
        current_t = t[i - 1]
        current_state = states[:, i - 1]

        # Find the closest voltage value from real data
        idx = np.argmin(np.abs(time_data - current_t))
        vm = voltage_data[idx]

        # RK4 integration
        k1 = dynamics_step(current_state, current_t, vm)
        k2 = dynamics_step(current_state + 0.5 * dt * k1, current_t + 0.5 * dt, vm)
        k3 = dynamics_step(current_state + 0.5 * dt * k2, current_t + 0.5 * dt, vm)
        k4 = dynamics_step(current_state + dt * k3, current_t + dt, vm)

        # Update state
        new_state = current_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[:, i] = enforce_theta_limits(new_state)

    return t, states


# -------------------- MAIN COMPARISON SCRIPT --------------------
def process_dataset(file_name):
    """Process a single dataset file and return simulation results and errors"""
    print(f"\nProcessing dataset: {file_name}")

    try:
        # Load real data
        real_data = pd.read_csv(file_name)

        # Convert time to seconds from milliseconds and set relative to start time
        real_data_time = (real_data['time'] - real_data['time'].iloc[0]) / 1000.0
        real_data_voltage = real_data['voltage'].values

        # Convert angles from degrees to radians for comparison
        real_data_angle = np.radians(real_data['position'].values)  # Arm angle (theta)
        real_data_unwrapped_angle = np.radians(real_data['unwrapped_pendulum_angle'].values)  # Pendulum angle (alpha)

        # Calculate derivatives (velocities) from real data
        dt_real = np.diff(real_data_time)
        real_data_angle_dot = np.zeros_like(real_data_angle)
        real_data_angle_dot[1:] = np.diff(real_data_angle) / dt_real

        real_data_unwrapped_angle_dot = np.zeros_like(real_data_unwrapped_angle)
        real_data_unwrapped_angle_dot[1:] = np.diff(real_data_unwrapped_angle) / dt_real

        print(f"Real data loaded successfully.")
        print(f"Real data time range: {real_data_time.min():.2f}s to {real_data_time.max():.2f}s")

        # Simulation parameters
        t_span = (0.0, real_data_time.max())  # Match real data time range
        dt = 0.001  # 1ms timestep (1000Hz) - can be adjusted for accuracy

        # Use the initial conditions from real data
        initial_theta = real_data_angle[0]
        initial_alpha = real_data_unwrapped_angle[0]
        initial_theta_dot = real_data_angle_dot[0]
        initial_alpha_dot = real_data_unwrapped_angle_dot[0]

        initial_state = np.array([initial_theta, initial_alpha, initial_theta_dot, initial_alpha_dot])

        print("Running simulation with real voltage data...")
        t_sim, states_sim = rk4_integration(
            initial_state, t_span, dt, real_data_voltage, real_data_time
        )

        # Extract results
        theta_sim = states_sim[0]  # Arm angle from simulation
        alpha_sim = states_sim[1]  # Pendulum angle from simulation

        # Calculate normalized angles for visualization
        alpha_normalized_sim = np.zeros(len(alpha_sim))
        for i in range(len(alpha_sim)):
            alpha_normalized_sim[i] = alpha_sim[i]

        # Calculate errors between simulation and real data
        # Interpolate simulation results to match real data time points
        from scipy.interpolate import interp1d

        # Interpolate simulation results
        theta_interp = interp1d(t_sim, theta_sim, bounds_error=False, fill_value="extrapolate")
        alpha_interp = interp1d(t_sim, alpha_normalized_sim, bounds_error=False, fill_value="extrapolate")

        # Calculate errors at real data time points
        theta_error = theta_interp(real_data_time) - real_data_angle
        alpha_error = alpha_interp(real_data_time) - real_data_unwrapped_angle

        # Calculate RMSE
        theta_rmse = np.sqrt(np.mean(theta_error ** 2))
        alpha_rmse = np.sqrt(np.mean(alpha_error ** 2))

        print(f"Simulation with real voltage - Theta RMSE: {theta_rmse:.4f} rad, Alpha RMSE: {alpha_rmse:.4f} rad")

        # Prepare results to return
        results = {
            'real_data_time': real_data_time,
            'real_data_angle': real_data_angle,
            'real_data_unwrapped_angle': real_data_unwrapped_angle,
            't_sim': t_sim,
            'theta_sim': theta_sim,
            'alpha_normalized_sim': alpha_normalized_sim,
            'theta_error': theta_error,
            'alpha_error': alpha_error,
            'theta_rmse': theta_rmse,
            'alpha_rmse': alpha_rmse
        }

        return results

    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return None


def create_comparison_plots(results, dataset_name):
    """Create comparison plots for a dataset"""
    if results is None:
        print(f"Skipping plot generation for {dataset_name} due to processing error.")
        return

    print(f"Generating comparison plots for {dataset_name}...")
    plt.figure(figsize=(16, 16))

    # Plot 1: Arm Angle (Theta) Comparison
    plt.subplot(4, 1, 1)
    plt.plot(results['real_data_time'], results['real_data_angle'], 'k-', linewidth=2, label='Real Data')
    plt.plot(results['t_sim'], results['theta_sim'], 'r-', linewidth=1.5, label='Simulation')
    plt.axhline(y=THETA_MAX, color='g', linestyle='-.', alpha=0.5, label='Theta Limits')
    plt.axhline(y=THETA_MIN, color='g', linestyle='-.', alpha=0.5)
    plt.ylabel('Arm angle (rad)')
    plt.title(f'Comparison of Arm Angle (Theta) - {dataset_name}')
    plt.legend()
    plt.grid(True)

    # Plot 2: Pendulum Angle (Alpha) Comparison
    plt.subplot(4, 1, 2)
    plt.plot(results['real_data_time'], results['real_data_unwrapped_angle'], 'k-', linewidth=2, label='Real Data')
    plt.plot(results['t_sim'], results['alpha_normalized_sim'], 'r-', linewidth=1.5, label='Simulation')
    plt.axhline(y=0, color='g', linestyle='-.', alpha=0.5, label='Upright Position')
    plt.ylabel('Pendulum angle (rad)')
    plt.title(f'Comparison of Pendulum Angle (Alpha) - {dataset_name}')
    plt.legend()
    plt.grid(True)

    # Plot 3: Arm Angle Error
    plt.subplot(4, 1, 3)
    plt.plot(results['real_data_time'], results['theta_error'], 'r-', linewidth=1.5,
             label=f'Simulation Error (RMSE: {results["theta_rmse"]:.4f})')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.ylabel('Arm angle error (rad)')
    plt.title(f'Arm Angle (Theta) Error - {dataset_name}')
    plt.legend()
    plt.grid(True)

    # Plot 4: Pendulum Angle Error
    plt.subplot(4, 1, 4)
    plt.plot(results['real_data_time'], results['alpha_error'], 'r-', linewidth=1.5,
             label=f'Simulation Error (RMSE: {results["alpha_rmse"]:.4f})')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Pendulum angle error (rad)')
    plt.title(f'Pendulum Angle (Alpha) Error - {dataset_name}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the figure
    output_filename = f"pendulum_simulation_vs_real_data_{dataset_name}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{output_filename}'")


def main():
    start_time = time()
    print("Starting pendulum simulation vs real data comparison for multiple datasets...")

    # List of datasets to process
    datasets = ['df1.csv', 'df2.csv', 'df3.csv', 'df4.csv', 'df5.csv', 'df6.csv']

    """datasets = ['processed_qube_data_20250329_111906.csv',
                'processed_qube_data_20250329_113057.csv',
                'processed_qube_data_20250329_122651.csv',
                'processed_qube_data_20250329_132333.csv',
                'processed_qube_data_20250329_132635.csv',
                'processed_qube_data_20250329_145238.csv']"""

    # Summary data for final report
    summary_data = []

    # Process each dataset
    for dataset in datasets:
        dataset_name = dataset.split('.')[0]  # Remove file extension
        results = process_dataset(dataset)

        if results:
            # Create plots for this dataset
            create_comparison_plots(results, dataset_name)

            # Add to summary
            summary_data.append({
                'dataset': dataset_name,
                'theta_rmse': results['theta_rmse'],
                'alpha_rmse': results['alpha_rmse']
            })

    # Create summary plot
    if summary_data:
        print("\nGenerating summary comparison...")
        datasets = [data['dataset'] for data in summary_data]
        theta_rmses = [data['theta_rmse'] for data in summary_data]
        alpha_rmses = [data['alpha_rmse'] for data in summary_data]

        plt.figure(figsize=(12, 8))

        x = np.arange(len(datasets))
        width = 0.35

        plt.bar(x - width / 2, theta_rmses, width, label='Arm Angle (Theta) RMSE')
        plt.bar(x + width / 2, alpha_rmses, width, label='Pendulum Angle (Alpha) RMSE')

        plt.xlabel('Dataset')
        plt.ylabel('RMSE (radians)')
        plt.title('Simulation Error Comparison Across Datasets')
        plt.xticks(x, datasets)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("pendulum_simulation_error_summary.png", dpi=300)
        print("Summary plot saved as 'pendulum_simulation_error_summary.png'")

        # Print summary table
        print("\nSummary of simulation errors across datasets:")
        print("=" * 60)
        print(f"{'Dataset':<10} {'Arm Angle RMSE':<20} {'Pendulum Angle RMSE':<20}")
        print("-" * 60)
        for data in summary_data:
            print(f"{data['dataset']:<10} {data['theta_rmse']:<20.4f} {data['alpha_rmse']:<20.4f}")
        print("=" * 60)

    sim_time = time() - start_time
    print(f"\nAnalysis completed in {sim_time:.2f} seconds!")


if __name__ == "__main__":
    main()