import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import numba as nb
from scipy import signal

# System Parameters from the system identification document (Table 3)
Rm = 8.94  # Motor resistance (Ohm)
Km = 0.0431  # Motor back-emf constant
Jm = 6e-5  # Total moment of inertia acting on motor shaft (kg·m^2)
bm = 3e-4  # Viscous damping coefficient (Nm/rad/s)
DA = 3e-4  # Damping coefficient of pendulum arm (Nm/rad/s)
DL = 5e-4  # Damping coefficient of pendulum link (Nm/rad/s)
mA = 0.053  # Weight of pendulum arm (kg)
mL = 0.024  # Weight of pendulum link (kg)
LA = 0.086  # Length of pendulum arm (m)
LL = 0.128  # Length of pendulum link (m)
JA = 5.72e-5  # Inertia moment of pendulum arm (kg·m^2)
JL = 1.31e-4  # Inertia moment of pendulum link (kg·m^2)
g = 9.81  # Gravity constant (m/s^2)

# Pre-compute constants for optimization
half_mL_LL_g = 0.5 * mL * LL * g
half_mL_LL_LA = 0.5 * mL * LL * LA
quarter_mL_LL_squared = 0.25 * mL * LL ** 2
Mp_g_Lp = mL * g * LL
Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia (kg·m²)

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
    if -0.2 <= vm <= 0.2:
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
    """Dynamics calculation for pendulum system"""
    theta_m, theta_L, theta_m_dot, theta_L_dot = state[0], state[1], state[2], state[3]

    # Check theta limits - implement hard stops
    if (theta_m >= THETA_MAX and theta_m_dot > 0) or (theta_m <= THETA_MIN and theta_m_dot < 0):
        theta_m_dot = 0.0  # Stop the arm motion at the limits

    # Apply dead zone and calculate motor torque
    vm = apply_voltage_deadzone(vm)

    # Motor torque calculation
    im = (vm - Km * theta_m_dot) / Rm
    Tm = Km * im

    # Equations of motion coefficients
    # For theta_m equation:
    M11 = mL * LA ** 2 + quarter_mL_LL_squared - quarter_mL_LL_squared * np.cos(theta_L) ** 2 + JA
    M12 = -half_mL_LL_LA * np.cos(theta_L)
    C1 = 0.5 * mL * LL ** 2 * np.sin(theta_L) * np.cos(theta_L) * theta_m_dot * theta_L_dot
    C2 = half_mL_LL_LA * np.sin(theta_L) * theta_L_dot ** 2

    # For theta_L equation:
    M21 = -half_mL_LL_LA * np.cos(theta_L)
    M22 = JL + quarter_mL_LL_squared
    C3 = -quarter_mL_LL_squared * np.cos(theta_L) * np.sin(theta_L) * theta_m_dot ** 2
    G = half_mL_LL_g * np.sin(theta_L)

    # Calculate determinant for matrix inversion
    det_M = M11 * M22 - M12 * M21

    # Handle near-singular matrix
    if abs(det_M) < 1e-10:
        theta_m_ddot = 0
        theta_L_ddot = 0
    else:
        # Right-hand side of equations
        RHS1 = Tm - C1 - C2 - DA * theta_m_dot
        RHS2 = -G - DL * theta_L_dot - C3

        # Solve for accelerations
        theta_m_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
        theta_L_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M

    return np.array([theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot])


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
def main():
    start_time = time()
    print("Starting pendulum simulation vs real data comparison...")

    # Load real data
    real_data = pd.read_csv('df3.csv')

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

    print("Real data loaded successfully.")
    print(f"Real data time range: {real_data_time.min():.2f}s to {real_data_time.max():.2f}s")

    # Simulation parameters
    t_span = (0.0, real_data_time.max())  # Match real data time range
    dt = 0.001  # 10ms timestep (100Hz) - can be adjusted for accuracy

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

    # Plot results
    print("Generating comparison plots...")
    plt.figure(figsize=(16, 16))

    # Plot 1: Arm Angle (Theta) Comparison
    plt.subplot(4, 1, 1)
    plt.plot(real_data_time, real_data_angle, 'k-', linewidth=2, label='Real Data')
    plt.plot(t_sim, theta_sim, 'r-', linewidth=1.5, label='Simulation')
    plt.axhline(y=THETA_MAX, color='g', linestyle='-.', alpha=0.5, label='Theta Limits')
    plt.axhline(y=THETA_MIN, color='g', linestyle='-.', alpha=0.5)
    plt.ylabel('Arm angle (rad)')
    plt.title('Comparison of Arm Angle (Theta)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Pendulum Angle (Alpha) Comparison
    plt.subplot(4, 1, 2)
    plt.plot(real_data_time, real_data_unwrapped_angle, 'k-', linewidth=2, label='Real Data')
    plt.plot(t_sim, alpha_normalized_sim, 'r-', linewidth=1.5, label='Simulation')
    plt.axhline(y=0, color='g', linestyle='-.', alpha=0.5, label='Upright Position')
    plt.ylabel('Pendulum angle (rad)')
    plt.title('Comparison of Pendulum Angle (Alpha)')
    plt.legend()
    plt.grid(True)

    # Plot 3: Arm Angle Error
    plt.subplot(4, 1, 3)
    plt.plot(real_data_time, theta_error, 'r-', linewidth=1.5, label=f'Simulation Error (RMSE: {theta_rmse:.4f})')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.ylabel('Arm angle error (rad)')
    plt.title('Arm Angle (Theta) Error')
    plt.legend()
    plt.grid(True)

    # Plot 4: Pendulum Angle Error
    plt.subplot(4, 1, 4)
    plt.plot(real_data_time, alpha_error, 'r-', linewidth=1.5, label=f'Simulation Error (RMSE: {alpha_rmse:.4f})')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Pendulum angle error (rad)')
    plt.title('Pendulum Angle (Alpha) Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("pendulum_simulation_vs_real_data.png", dpi=300)
    print("Plot saved as 'pendulum_simulation_vs_real_data.png'")

    sim_time = time() - start_time
    print(f"Analysis completed in {sim_time:.2f} seconds!")


if __name__ == "__main__":
    main()