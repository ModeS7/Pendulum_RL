import numpy as np
import matplotlib.pyplot as plt
from time import time
import numba as nb
from collections import deque

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
THETA_MIN = -2.2  # Minimum arm angle (radians)
THETA_MAX = 2.2  # Maximum arm angle (radians)

# Replace the existing cable_fourier list with a numpy array
# Format: (frequency, amplitude, phase) in rows
cable_fourier = np.array([
    [0.4998, 0.062, -1.033],
    [1.4988, 0.128, 0.829],
    [2.4988, 0.1464, -1.122],
    [3.4983, 0.067, 0.972],
    [4.4983, 0.0359, 0.401]
], dtype=np.float64)

# Global flag to enable/disable frequency-dependent damping
USE_FREQ_DAMPING = True
# Number of Fourier terms to use (1-5, or None for all)
NUM_FOURIER_TERMS = None

# Global state for frequency estimation (since numba doesn't support class instances)
# We'll use fixed-size arrays instead of deques for numba compatibility
HISTORY_SIZE = 10
theta_m_history = np.zeros(HISTORY_SIZE)
theta_L_history = np.zeros(HISTORY_SIZE)
time_history = np.zeros(HISTORY_SIZE)
history_index = 0
last_time = 0.0


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


@nb.njit(fastmath=True, cache=True)
def update_history(theta_m, theta_L, t, theta_m_history, theta_L_history,
                   time_history, history_index, last_time):
    """Update history arrays for frequency estimation and return updated state"""
    # Update our circular buffer
    theta_m_history[history_index] = theta_m
    theta_L_history[history_index] = theta_L
    time_history[history_index] = t

    # Update index for next entry
    new_history_index = (history_index + 1) % HISTORY_SIZE
    new_last_time = t

    return theta_m_history, theta_L_history, time_history, new_history_index, new_last_time


@nb.njit(fastmath=True, cache=True)
def estimate_frequency(theta_array, time_array):
    """
    Estimate oscillation frequency using zero-crossing detection
    """
    # Calculate mean of the signal
    mean_value = 0.0
    count = 0
    for i in range(HISTORY_SIZE):
        if abs(theta_array[i]) > 1e-10:  # Only count non-zero entries
            mean_value += theta_array[i]
            count += 1

    if count > 0:
        mean_value /= count
    else:
        return 1.0  # Default if no valid data

    # Find zero crossings (mean crossings)
    crossings = 0
    crossing_times = np.zeros(HISTORY_SIZE // 2)  # Maximum possible crossings
    crossing_count = 0

    # Get the first valid sign
    last_sign = 0
    for i in range(HISTORY_SIZE):
        if abs(theta_array[i]) > 1e-10:  # Only use non-zero entries
            diff = theta_array[i] - mean_value
            if abs(diff) > 1e-10:  # Avoid exact zeros
                last_sign = 1 if diff > 0 else -1
                break

    # Count crossings
    for i in range(1, HISTORY_SIZE):
        if abs(theta_array[i]) > 1e-10:  # Only use non-zero entries
            diff = theta_array[i] - mean_value
            if abs(diff) > 1e-10:  # Avoid exact zeros
                current_sign = 1 if diff > 0 else -1
                if current_sign != last_sign:
                    crossings += 1
                    if crossing_count < len(crossing_times):
                        crossing_times[crossing_count] = time_array[i]
                        crossing_count += 1
                    last_sign = current_sign

    # Calculate frequency from zero crossings
    if crossing_count >= 2:
        # Time between first and last crossing divided by number of half-periods
        period = (crossing_times[crossing_count - 1] - crossing_times[0]) / (crossing_count - 1) * 2
        if period > 0.01:  # Avoid division by very small numbers
            return 1.0 / period

    return 1.0  # Default frequency if estimation fails


@nb.njit(fastmath=True, cache=True)
def calc_frequency_damping(theta_m, theta_L, theta_m_dot, theta_L_dot, t, num_terms,
                          theta_m_history, theta_L_history, time_history,
                          history_index, last_time):
    """
    Calculate damping based on frequency content

    Returns:
    --------
    tuple: (motor_damping_force, link_damping_force, updated_history_state)
    """
    # Update history for frequency estimation
    theta_m_history, theta_L_history, time_history, history_index, last_time = update_history(
        theta_m, theta_L, t, theta_m_history, theta_L_history, time_history,
        history_index, last_time
    )

    # Default case - use standard damping
    if not USE_FREQ_DAMPING:
        return DA * theta_m_dot, DL * theta_L_dot, (theta_m_history, theta_L_history,
                                                   time_history, history_index, last_time)

    # Not enough history data yet
    valid_entries = min(history_index + 1, HISTORY_SIZE)
    if valid_entries < 3:
        return DA * theta_m_dot, DL * theta_L_dot, (theta_m_history, theta_L_history,
                                                   time_history, history_index, last_time)

    # Estimate frequencies
    motor_freq = estimate_frequency(theta_m_history, time_history)
    link_freq = estimate_frequency(theta_L_history, time_history)

    # Base damping coefficients
    motor_damping = DA
    link_damping = DL

    # Determine how many Fourier terms to use
    max_terms = cable_fourier.shape[0]  # Use shape[0] instead of len()
    if num_terms is None or num_terms > max_terms:
        num_terms = max_terms

    # Add frequency-dependent components
    for i in range(num_terms):
        # Access array elements by index instead of unpacking tuple
        freq = cable_fourier[i, 0]
        amp = cable_fourier[i, 1]
        phase = cable_fourier[i, 2]

        # Calculate Fourier terms
        motor_fourier = amp * np.sin(2 * np.pi * freq * t + phase)
        link_fourier = amp * np.sin(2 * np.pi * freq * t + phase)

        # Weight by frequency proximity (Gaussian weighting)
        motor_weight = np.exp(-2 * (motor_freq - freq) ** 2)
        link_weight = np.exp(-2 * (link_freq - freq) ** 2)

        # Add weighted contribution
        motor_damping += motor_fourier * motor_weight * 0.001  # Scale factor
        link_damping += link_fourier * link_weight * 0.001  # Scale factor

    # Ensure damping doesn't become negative (which would add energy)
    motor_damping = max(0.0001, motor_damping)
    link_damping = max(0.0001, link_damping)

    # Return forces AND updated history state
    return motor_damping * theta_m_dot, link_damping * theta_L_dot, (theta_m_history,
           theta_L_history, time_history, history_index, last_time)


# -------------------- CONTROL ALGORITHMS --------------------
@nb.njit(fastmath=True, cache=True)
def simple_bang_bang(t, theta, alpha, theta_dot, alpha_dot):
    """Ultra-fast bang-bang controller"""
    # Add theta limit avoidance to controller
    limit_margin = 0.6
    if theta > THETA_MAX - limit_margin and theta_dot > 0:
        return -max_voltage  # If close to upper limit, push back
    elif theta < THETA_MIN + limit_margin and theta_dot < 0:
        return max_voltage  # If close to lower limit, push back

    if alpha == 0:
        alpha = 0
    else:
        side = np.sin(alpha) / abs(np.sin(alpha))
    if alpha_dot == 0:
        alpha_dot = 0
    else:
        direction = alpha_dot / abs(alpha_dot)

    if theta > THETA_MAX - 0.6:
        direction = 1
    elif theta < THETA_MIN + 0.6:
        direction = -1

    if direction == 1:
        v = -max_voltage
    elif direction == -1:
        v = max_voltage
    else:
        v = 0

    return clip_value(v, -max_voltage, max_voltage)


@nb.njit(fastmath=True, cache=True)
def energy_control(t, theta, alpha, theta_dot, alpha_dot):
    alpha_norm = normalize_angle(alpha + np.pi)
    E = Mp_g_Lp * (1 - np.cos(alpha)) + 0.5 * Jp * alpha_dot ** 2
    E_ref = Mp_g_Lp  # Energy at upright position
    E_error = E - E_ref

    # Exponential penalty when approaching limits
    theta_penalty = 1.0
    limit_margin = 0.5  # How close to the limit before strong penalty
    if theta > THETA_MAX - limit_margin:
        limit_penalty = 10.0 * np.exp((theta - (THETA_MAX - limit_margin)) / limit_margin)
        theta_penalty += limit_penalty
    elif theta < THETA_MIN + limit_margin:
        limit_penalty = -10.0 * np.exp(((THETA_MIN + limit_margin) - theta) / limit_margin)
        theta_penalty += limit_penalty

    direction = 1.0 if np.cos(alpha_norm) * alpha_dot > 0 else -1.0
    u = 0.5 * E_error * direction - theta_penalty

    return clip_value(u, -max_voltage, max_voltage)


@nb.njit(fastmath=True, cache=True)
def lqr_balance(theta, alpha, theta_dot, alpha_dot):
    """Ultra-fast LQR controller with theta limits consideration"""
    alpha_upright = normalize_angle(alpha - np.pi)

    # Regular LQR control
    u = -(-5.0 * theta + 50.0 * alpha_upright - 1.5 * theta_dot + 8.0 * alpha_dot)

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
    elif Mp_g_Lp * (
            1 - np.cos(alpha)) + 0.5 * Jp * alpha_dot ** 2 < Mp_g_Lp * 1.1:  # Current energy vs energy threshold
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
def rk4_step(state, t, dt, control_func, history_state):
    """
    4th-order Runge-Kutta integrator step with theta limits enforcement
    """
    # Get control input based on current state
    control_output = control_func(t, state)
    vm = control_output[0]  # Extract control value only for dynamics

    # Apply limits to initial state
    state = enforce_theta_limits(state)

    # RK4 integration with limit enforcement at each step
    k1, history_state = dynamics_step(state, t, vm, history_state)

    # Apply limits after each partial step
    state_k2 = enforce_theta_limits(state + 0.5 * dt * k1)
    k2, history_state = dynamics_step(state_k2, t + 0.5 * dt, vm, history_state)

    state_k3 = enforce_theta_limits(state + 0.5 * dt * k2)
    k3, history_state = dynamics_step(state_k3, t + 0.5 * dt, vm, history_state)

    state_k4 = enforce_theta_limits(state + dt * k3)
    k4, history_state = dynamics_step(state_k4, t + dt, vm, history_state)

    # Apply limits to final integrated state
    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return enforce_theta_limits(new_state), history_state


@nb.njit(fastmath=True, cache=True)
def dynamics_step(state, t, vm, history_state):
    """Ultra-optimized dynamics calculation with frequency-dependent damping"""
    theta_m, theta_L, theta_m_dot, theta_L_dot = state[0], state[1], state[2], state[3]
    theta_m_history, theta_L_history, time_history, history_index, last_time = history_state

    # Check theta limits - implement hard stops
    if (theta_m >= THETA_MAX and theta_m_dot > 0) or (theta_m <= THETA_MIN and theta_m_dot < 0):
        theta_m_dot = 0.0  # Stop the arm motion at the limits

    # Apply dead zone and calculate motor torque
    vm = apply_voltage_deadzone(vm)

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
    M21 = half_mL_LL_LA * np.cos(theta_L)
    M22 = JL + quarter_mL_LL_squared
    C3 = -quarter_mL_LL_squared * np.cos(theta_L) * np.sin(theta_L) * theta_m_dot ** 2
    G = half_mL_LL_g * np.sin(theta_L)

    # Calculate frequency-dependent damping instead of constant damping
    motor_damping, link_damping, updated_history_state = calc_frequency_damping(
        theta_m, theta_L, theta_m_dot, theta_L_dot, t, NUM_FOURIER_TERMS,
        theta_m_history, theta_L_history, time_history, history_index, last_time
    )

    # Calculate determinant for matrix inversion
    det_M = M11 * M22 - M12 * M21

    # Handle near-singular matrix
    if abs(det_M) < 1e-10:
        theta_m_ddot = 0
        theta_L_ddot = 0
    else:
        # Right-hand side of equations
        RHS1 = Tm - C1 - C2 - motor_damping  # Use calculated motor damping
        RHS2 = -G - link_damping - C3  # Use calculated link damping

        # Solve for accelerations
        theta_m_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
        theta_L_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M

    return np.array([theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot]), updated_history_state


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

    # Initialize history state
    theta_m_history = np.zeros(HISTORY_SIZE)
    theta_L_history = np.zeros(HISTORY_SIZE)
    time_history = np.zeros(HISTORY_SIZE)
    history_index = 0
    last_time = 0.0
    history_state = (theta_m_history, theta_L_history, time_history, history_index, last_time)

    # Set initial state with limits applied
    states[:, 0] = enforce_theta_limits(state0)

    # Get initial control and controller mode
    control_output = control_func(t_start, states[:, 0])
    controls[0] = control_output[0]
    controller_modes[0] = control_output[1]

    # Integration loop
    for i in range(1, n_steps):
        states[:, i], history_state = rk4_step(states[:, i - 1], t[i - 1], dt, control_func, history_state)

        # Get control value and controller mode
        control_output = control_func(t[i], states[:, i])
        controls[i] = control_output[0]
        controller_modes[i] = control_output[1]

    return t, states, controls, controller_modes


# -------------------- MAIN SIMULATION --------------------
def main(use_freq_damping=True, num_fourier_terms=None):
    global USE_FREQ_DAMPING, NUM_FOURIER_TERMS

    # Set global parameters
    USE_FREQ_DAMPING = use_freq_damping
    NUM_FOURIER_TERMS = num_fourier_terms

    start_time = time()
    damping_mode = "frequency-dependent" if USE_FREQ_DAMPING else "standard"
    terms_desc = f" with {NUM_FOURIER_TERMS} terms" if USE_FREQ_DAMPING and NUM_FOURIER_TERMS is not None else ""
    print(f"Starting pendulum simulation with {damping_mode} damping{terms_desc}...")

    # Simulation parameters
    t_span = (0.0, 15.0)  # 15 seconds of simulation
    dt = 0.02  # 20ms timestep (50Hz)

    # Initial conditions
    initial_state = np.array([0.0, 0.1, 0.0, 0.0])  # [theta, alpha, theta_dot, alpha_dot]

    print("=" * 50)
    print(f"STARTING SIMULATION WITH THETA LIMITS: [{THETA_MIN}, {THETA_MAX}]")
    print(f"DAMPING MODEL: {damping_mode}{terms_desc}")
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
    title = f'Inverted Pendulum Control with {damping_mode.capitalize()} Damping{terms_desc}'
    plt.title(f'{title} (Simulation time: {sim_time:.2f}s)')
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
    plt.yticks([0.25, 0.5, 0.75], ['Emergency', 'Bang-Bang', 'LQR/Energy'])
    plt.legend(title='Controller Modes', loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    filename = f"pendulum_{damping_mode}_damping"
    if USE_FREQ_DAMPING and NUM_FOURIER_TERMS is not None:
        filename += f"_{NUM_FOURIER_TERMS}_terms"
    filename += ".png"
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")
    plt.show()

    print("=" * 50)
    print(f"PROGRAM EXECUTION COMPLETE IN {time() - start_time:.2f} SECONDS")
    print("=" * 50)

    return {
        "simulation_time": sim_time,
        "balanced_time": balanced_time,
        "inversion_success": inversion_success,
        "limit_hits": limit_hits
    }


def compare_damping_models():
    """Compare performance of different damping models"""
    print("\n" + "=" * 60)
    print("COMPARING DAMPING MODELS".center(60))
    print("=" * 60 + "\n")

    # Test configurations
    configs = [
        {"name": "Standard Viscous Damping", "freq_damping": False},
        {"name": "Frequency Damping (1 term)", "freq_damping": True, "terms": 1},
        {"name": "Frequency Damping (3 terms)", "freq_damping": True, "terms": 3},
        {"name": "Frequency Damping (All 5 terms)", "freq_damping": True, "terms": 5}
    ]

    # Run simulations and collect results
    results = []
    for config in configs:
        print(f"\nRunning simulation with {config['name']}...")
        result = main(
            use_freq_damping=config["freq_damping"],
            num_fourier_terms=config.get("terms", None)
        )
        results.append({
            "name": config["name"],
            "simulation_time": result["simulation_time"],
            "balanced_time": result["balanced_time"],
            "inversion_success": result["inversion_success"],
            "limit_hits": result["limit_hits"]
        })

    # Summarize results
    print("\n" + "=" * 80)
    print("DAMPING MODEL COMPARISON SUMMARY".center(80))
    print("=" * 80)

    # Print table header
    print(f"\n{'Model':<30} | {'Sim Time (s)':<12} | {'Balanced (s)':<12} | {'Success':<8} | {'Limit Hits':<10}")
    print("-" * 30 + "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 8 + "-+-" + "-" * 10)

    # Print results
    for r in results:
        print(
            f"{r['name']:<30} | {r['simulation_time']:<12.3f} | {r['balanced_time']:<12.3f} | {r['inversion_success']:<8} | {r['limit_hits']:<10}")

    # Create a bar chart comparing simulation times
    plt.figure(figsize=(12, 8))

    # Plot simulation times
    plt.subplot(2, 1, 1)
    names = [r["name"] for r in results]
    sim_times = [r["simulation_time"] for r in results]
    plt.bar(names, sim_times)
    plt.ylabel("Simulation Time (s)")
    plt.title("Computational Performance Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    # Plot balanced times
    plt.subplot(2, 1, 2)
    balanced_times = [r["balanced_time"] for r in results]
    plt.bar(names, balanced_times)
    plt.ylabel("Time Pendulum Stayed Balanced (s)")
    plt.title("Control Performance Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig("damping_model_comparison.png")
    print("\nComparison chart saved as 'damping_model_comparison.png'")
    plt.show()

if __name__ == "__main__":
    # Run main simulation
    main(use_freq_damping=True, num_fourier_terms=None)

    # Compare damping models
    compare_damping_models()
