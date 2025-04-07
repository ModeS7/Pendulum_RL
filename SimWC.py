import numpy as np
import matplotlib.pyplot as plt
from time import time
import numba as nb

from QUBE_PYTHON.logger import directory

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

# Define controller mode constants
EMERGENCY_MODE = 0
BANGBANG_MODE = 1
LQR_MODE = 2
ENERGY_MODE = 3

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



# -------------------- CONTROL ALGORITHMS --------------------
@nb.njit(fastmath=True, cache=True)
def simple_bang_bang(t, theta, alpha, theta_dot, alpha_dot):
    """Ultra-fast bang-bang controller for inverted pendulum swing-up"""
    # First, handle limit avoidance with highest priority
    limit_margin = 1.5
    if theta > THETA_MAX - limit_margin and theta_dot > 0:
        return -max_voltage  # If close to upper limit, push back
    elif theta < THETA_MIN + limit_margin and theta_dot < 0:
        return max_voltage  # If close to lower limit, push back

    pos_vel_same_sign = alpha * alpha_dot > 0
    if pos_vel_same_sign:
        # Apply torque against position to pump energy
        if alpha < 0:
            return -max_voltage
        else:
            return max_voltage
    else:
        # Apply torque with position
        if alpha < 0:
            return max_voltage
        else:
            return -max_voltage


@nb.njit(fastmath=True, cache=True)
def energy_control(t, theta, alpha, theta_dot, alpha_dot):
    """Improved energy-based swing-up controller with enhanced limit avoidance"""
    # Calculate current energy and reference energy
    E_current = Mp_g_Lp * (1 - np.cos(alpha)) + 0.5 * Jp * alpha_dot ** 2
    E_ref = 2 * Mp_g_Lp  # Energy at upright position (2*m*g*L)
    E_error = E_ref - E_current  # Positive when we need to add energy

    # Use standard energy pumping formula with sign(alpha_dot * sin(alpha))
    # This determines when to apply torque to efficiently add/remove energy
    pump_direction = np.sign(alpha_dot * np.sin(alpha))

    # Adaptive gain - use smaller gain when close to target energy for smoother control
    k_energy = 0.5  # Base gain
    if abs(E_error) < 0.3 * Mp_g_Lp:
        k_energy = 0.3  # More precise control when close to target

    # Energy pumping control
    u_energy = k_energy * E_error * pump_direction

    # Enhanced limit avoidance with velocity consideration
    # Use larger margin when moving quickly to account for momentum
    base_margin = 0.5
    velocity_factor = min(1.0, abs(theta_dot) / 2.0)
    dynamic_margin = base_margin * (1.0 + velocity_factor)

    # Calculate limit avoidance control
    u_limit = 0.0
    if theta > THETA_MAX - dynamic_margin:
        # Stronger repulsion from upper limit when moving quickly
        distance_ratio = (theta - (THETA_MAX - dynamic_margin)) / dynamic_margin
        u_limit = -12.0 * np.exp(distance_ratio) * (1.0 + velocity_factor)
    elif theta < THETA_MIN + dynamic_margin:
        # Stronger repulsion from lower limit when moving quickly
        distance_ratio = ((THETA_MIN + dynamic_margin) - theta) / dynamic_margin
        u_limit = 12.0 * np.exp(distance_ratio) * (1.0 + velocity_factor)

    # Add damping when pendulum has high velocity to prevent wild swings
    u_damping = -0.1 * alpha_dot if abs(alpha_dot) > 5.0 else 0.0

    # Combine all control components
    u = u_energy + u_limit + u_damping

    return clip_value(u, -max_voltage, max_voltage)

@nb.njit(fastmath=True, cache=True)
def lqr_balance(theta, alpha, theta_dot, alpha_dot):
    """Ultra-fast LQR controller with theta limits consideration"""
    alpha_upright = normalize_angle(alpha - np.pi)

    # Regular LQR control
    #u = (2.0 * theta + 10.0 * alpha_upright + 1.5 * theta_dot + 8.0 * alpha_dot)  # 7v
    u = (2.0 * theta + 50.0 * alpha_upright + 1.5 * theta_dot + 8.0 * alpha_dot) # 10v

    # Add limit avoidance term
    limit_margin = 0.3
    if theta > THETA_MAX - limit_margin:
        # Add strong negative control to avoid upper limit
        avoid_factor = 20.0 * (theta - (THETA_MAX - limit_margin)) / limit_margin
        u -= avoid_factor
    elif theta < THETA_MIN + limit_margin:
        # Add strong positive control to avoid lower limit
        avoid_factor = 20.0 * ((THETA_MIN + limit_margin) - theta) / limit_margin
        u += avoid_factor

    return clip_value(u, -max_voltage, max_voltage)

@nb.njit(fastmath=True, cache=True)
def control_decision(t, state):
    """Combined controller with controller mode tracking"""
    theta, alpha, theta_dot, alpha_dot = state[0], state[1], state[2], state[3]
    alpha_norm = normalize_angle(alpha + np.pi)

    # Emergency limit handling - override all other controllers
    if (theta >= THETA_MAX and theta_dot > 0) or (theta <= THETA_MIN and theta_dot < 0):
        control_value = -max_voltage if theta_dot > 0 else max_voltage
        return control_value, EMERGENCY_MODE
    elif Mp_g_Lp * (1 - np.cos(alpha)) + 0.5 * Jp * alpha_dot ** 2 < Mp_g_Lp * 1.1:  # Current energy vs energy threshold
        control_value = simple_bang_bang(t, theta, alpha, theta_dot, alpha_dot)
        return control_value, BANGBANG_MODE
    elif abs(alpha_norm) < 0.3:
        control_value = lqr_balance(theta, alpha, theta_dot, alpha_dot)
        return control_value, LQR_MODE
    else:
        control_value = energy_control(t, theta, alpha, theta_dot, alpha_dot)
        return control_value, ENERGY_MODE



# -------------------- CUSTOM INTEGRATOR --------------------
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

@nb.njit(fastmath=True, parallel=False, cache=True)
def rk4_step(state, t, dt, control_func):
    """
    4th-order Runge-Kutta integrator step with theta limits enforcement
    """
    # Get control input based on current state
    control_output = control_func(t, state)
    vm = control_output[0]  # Extract control value only for dynamics

    # Apply limits to initial state
    state = enforce_theta_limits(state)

    # RK4 integration with limit enforcement at each step
    k1 = dynamics_step(state, t, vm)

    # Apply limits after each partial step
    state_k2 = enforce_theta_limits(state + 0.5 * dt * k1)
    k2 = dynamics_step(state_k2, t + 0.5 * dt, vm)

    state_k3 = enforce_theta_limits(state + 0.5 * dt * k2)
    k3 = dynamics_step(state_k3, t + 0.5 * dt, vm)

    state_k4 = enforce_theta_limits(state + dt * k3)
    k4 = dynamics_step(state_k4, t + dt, vm)

    # Apply limits to final integrated state
    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return enforce_theta_limits(new_state)

@nb.njit(fastmath=True, cache=True)
def dynamics_step(state, t, vm):
    """Ultra-optimized dynamics calculation with theta limits"""
    theta_m, theta_L, theta_m_dot, theta_L_dot = state[0], state[1], state[2], state[3]

    # Check theta limits - implement hard stops
    if (theta_m >= THETA_MAX and theta_m_dot > 0) or (theta_m <= THETA_MIN and theta_m_dot < 0):
        theta_m_dot = 0.0  # Stop the arm motion at the limits

    # Apply dead zone and calculate motor torque
    apply_voltage_deadzone(vm)

    # Motor torque calculation
    im = (vm - Km * theta_m_dot) / Rm
    Tm = Km * im

    # Equations of motion coefficients from Eq. (9) in paper
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

@nb.njit(fastmath=True, parallel=False, cache=True)
def custom_integrate(state0, t_span, dt, control_func):
    """
    Custom integrator with theta limits enforcement and controller logging
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    # Pre-allocate arrays for results
    t = np.linspace(t_start, t_end, n_steps)
    states = np.zeros((4, n_steps))
    controls = np.zeros(n_steps)
    controller_modes = np.zeros(n_steps, dtype=np.int32)  # Track active controller

    # Set initial state with limits applied
    states[:, 0] = enforce_theta_limits(state0)

    # Get initial control and controller mode
    control_output = control_func(t_start, states[:, 0])
    controls[0] = control_output[0]
    controller_modes[0] = control_output[1]

    # Integration loop
    for i in range(1, n_steps):
        states[:, i] = rk4_step(states[:, i - 1], t[i - 1], dt, control_func)

        # Get control value and controller mode
        control_output = control_func(t[i], states[:, i])
        controls[i] = control_output[0]
        controller_modes[i] = control_output[1]

    return t, states, controls, controller_modes



# -------------------- MAIN SIMULATION --------------------
def main():
    start_time = time()
    print("Starting pendulum simulation with theta limits...")

    # Simulation parameters
    t_span = (0.0, 15.0)  # 10 seconds of simulation
    dt = 0.0115  # 11.5ms timestep (87Hz)

    # Initial conditions
    initial_state = np.array([0.0, 0.1, 0.0, 0.0])  # [theta, alpha, theta_dot, alpha_dot]

    print("=" * 50)
    print(f"STARTING SIMULATION WITH THETA LIMITS: [{THETA_MIN}, {THETA_MAX}]")
    print("=" * 50)

    # Custom integration with controller mode tracking
    t, states, controls, controller_modes = custom_integrate(initial_state, t_span, dt, control_decision)

    sim_time = time() - start_time
    print(f"Simulation completed in {sim_time:.2f} seconds!")

    # Extract results
    theta = states[0]
    alpha = states[1]
    theta_dot = states[2]
    alpha_dot = states[3]

    # Normalize alpha for visualization
    alpha_normalized = np.zeros(len(alpha))
    for i in range(len(alpha)):
        alpha_normalized[i] = normalize_angle(alpha[i] + np.pi)

    # Calculate performance metrics
    print("Calculating performance metrics...")
    inversion_success = False
    balanced_time = 0.0
    num_upright_points = 0
    limit_hits = 0

    # Count controller mode statistics
    emergency_time = 0.0
    bangbang_time = 0.0
    lqr_time = 0.0
    energy_time = 0.0

    for i in range(len(t)):
        # Check if pendulum is close to upright
        if abs(alpha_normalized[i]) < 0.17:  # about 10 degrees
            balanced_time += t[i] - t[i - 1] if i > 0 else 0
            num_upright_points += 1
            inversion_success = True

        # Count limit hits
        if abs(theta[i] - THETA_MAX) < 0.01 or abs(theta[i] - THETA_MIN) < 0.01:
            limit_hits += 1

        # Tally controller mode usage time
        dt_i = t[i] - t[i - 1] if i > 0 else 0
        if controller_modes[i] == EMERGENCY_MODE:
            emergency_time += dt_i
        elif controller_modes[i] == BANGBANG_MODE:
            bangbang_time += dt_i
        elif controller_modes[i] == LQR_MODE:
            lqr_time += dt_i
        elif controller_modes[i] == ENERGY_MODE:
            energy_time += dt_i

    print(f"Did pendulum reach inverted position? {inversion_success}")
    print(f"Time spent balanced (approximately): {balanced_time:.2f} seconds")
    print(f"Number of data points with pendulum upright: {num_upright_points}")
    print(f"Max arm angle: {np.max(np.abs(theta)):.2f} rad")
    print(f"Theta limit hits: {limit_hits} times")
    print(f"Max pendulum angular velocity: {np.max(np.abs(alpha_dot)):.2f} rad/s")
    final_angle_deg = abs(alpha_normalized[-1]) * 180 / np.pi
    print(f"Final pendulum angle from vertical: {abs(alpha_normalized[-1]):.2f} rad ({final_angle_deg:.1f} degrees)")
    print("\nController usage statistics:")
    print(f"- Emergency limit control: {emergency_time:.2f}s ({emergency_time / t_span[1] * 100:.1f}%)")
    print(f"- Bang-bang control: {bangbang_time:.2f}s ({bangbang_time / t_span[1] * 100:.1f}%)")
    print(f"- LQR balance control: {lqr_time:.2f}s ({lqr_time / t_span[1] * 100:.1f}%)")
    print(f"- Energy swing-up control: {energy_time:.2f}s ({energy_time / t_span[1] * 100:.1f}%)")

    # Plot results
    print("Generating plots...")
    plt.close()
    plt.figure(figsize=(14, 16))  # Make figure taller for 4 subplots

    # Plot arm angle
    plt.subplot(4, 1, 1)
    plt.plot(t, theta, 'b-')
    plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
    plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
    plt.ylabel('Arm angle (rad)')
    plt.title(f'Inverted Pendulum Control with Theta Limits (Simulation time: {sim_time:.2f}s)')
    plt.legend()
    plt.grid(True)

    # Plot pendulum angle
    plt.subplot(4, 1, 2)
    plt.scatter(t, alpha_normalized)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Line at upright position
    plt.ylabel('Pendulum angle (rad)')
    plt.grid(True)

    # Plot control signal
    plt.subplot(4, 1, 3)
    plt.plot(t, controls, 'g-')
    plt.ylabel('Control voltage (V)')
    plt.grid(True)

    # Plot controller mode
    plt.subplot(4, 1, 4)
    colors = ['red', 'orange', 'green', 'blue']
    labels = ['Emergency', 'Bang-Bang', 'LQR', 'Energy']

    # Create colored regions for different controller modes
    for i in range(4):
        mode_mask = controller_modes == i
        if np.any(mode_mask):
            plt.fill_between(t, 0, 1, where=mode_mask, color=colors[i], alpha=0.3, label=labels[i])

    # Create custom colored bars for controller transitions
    prev_mode = controller_modes[0]
    mode_changes = []
    for i in range(1, len(t)):
        if controller_modes[i] != prev_mode:
            mode_changes.append((t[i], prev_mode, controller_modes[i]))
            prev_mode = controller_modes[i]

    # Mark transition points with vertical lines
    for tc, _, _ in mode_changes:
        plt.axvline(x=tc, color='black', linestyle='-', alpha=0.2)

    # Add legend and labels
    plt.xlabel('Time (s)')
    plt.ylabel('Controller Mode')
    plt.yticks([])
    plt.legend(title='Controller Modes', loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("pendulum_with_controller_modes.png")
    print("Plot saved as 'pendulum_with_controller_modes.png'")
    plt.show()

    print("=" * 50)
    print(f"PROGRAM EXECUTION COMPLETE IN {time() - start_time:.2f} SECONDS")
    print("=" * 50)


if __name__ == "__main__":
    main()