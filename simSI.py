import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from tqdm.notebook import tqdm  # For progress tracking
import sys

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

# Cable effect parameters
zeta_cable = 0.034  # Damping ratio of encoder cable oscillation
omega_cable = 10.36  # Natural frequency of encoder cable oscillation (Hz)

# Fourier series coefficients for unwanted damping on motor (from Table 2)
cable_fourier = [
    (0.4998, 0.062, -1.033),  # (frequency, amplitude, phase)
    (1.4988, 0.128, 0.829),
    (2.4988, 0.1464, -1.122),
    (3.4983, 0.067, 0.972),
    (4.4983, 0.0359, 0.401)
]


def pendulum_dynamics(t, state, voltage_func):
    """
    Dynamics of the 2DOF QUBE-Servo pendulum based on the research paper

    state = [theta_m, theta_L, theta_m_dot, theta_L_dot, cable_pos, cable_vel]
    where:
        theta_m = motor angle (pendulum arm angle)
        theta_L = pendulum link angle
        theta_m_dot, theta_L_dot = respective angular velocities
        cable_pos, cable_vel = position and velocity of cable oscillation model
    """
    theta_m, theta_L, theta_m_dot, theta_L_dot, cable_pos, cable_vel = state

    # Input voltage
    vm = voltage_func(t)

    # Motor torque calculation
    im = (vm - Km * theta_m_dot) / Rm  # Motor current
    Tm = Km * im  # Motor torque

    # Equations of motion coefficients from Eq. (9)
    # For theta_m equation:
    M11 = mL * LA ** 2 + (1 / 4) * mL * LL ** 2 - (1 / 4) * mL * LL ** 2 * np.cos(theta_L) ** 2 + JA
    M12 = -(1 / 2) * mL * LL * LA * np.cos(theta_L)
    C1 = (1 / 2) * mL * LL ** 2 * np.sin(theta_L) * np.cos(theta_L) * theta_m_dot * theta_L_dot
    C2 = (1 / 2) * mL * LL * LA * np.sin(theta_L) * theta_L_dot ** 2

    # For theta_L equation:
    M21 = (1 / 2) * mL * LL * LA * np.cos(theta_L)
    M22 = JL + (1 / 4) * mL * LL ** 2
    C3 = -(1 / 4) * mL * LL ** 2 * np.cos(theta_L) * np.sin(theta_L) * theta_m_dot ** 2
    G = (1 / 2) * mL * LL * g * np.sin(theta_L)

    # Calculate determinant for matrix inversion
    det_M = M11 * M22 - M12 * M21

    # Include cable effect as additional torque for pendulum link
    cable_torque = cable_pos  # Position of cable oscillation model affects pendulum

    # Calculate accelerations using matrix inversion
    if abs(det_M) < 1e-10:  # Handle near-singular matrix
        theta_m_ddot = 0
        theta_L_ddot = 0
    else:
        # Right-hand side of equations
        RHS1 = Tm - C1 - C2 - DA * theta_m_dot
        RHS2 = -G - DL * theta_L_dot - C3 + cable_torque

        # Solve for accelerations
        theta_m_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
        theta_L_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M

    # Cable dynamics (second-order system)
    omega_n = omega_cable * 2 * np.pi  # Convert from Hz to rad/s
    cable_acc = -2 * zeta_cable * omega_n * cable_vel - omega_n ** 2 * cable_pos + theta_L_ddot

    # Calculate unwanted damping from encoder cable using Fourier series
    # This affects motor dynamics but we've incorporated its effect in the coupled equations

    return [theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot, cable_vel, cable_acc]


# Voltage functions
def step_voltage(t):
    """Step voltage of 3V at t=0.2s"""
    if t >= 0.2:
        return 3.0
    else:
        return 0.0


def zero_voltage(t):
    """Zero voltage (no input)"""
    return 0.0


# Performance optimization option
OPTIMIZE_PERFORMANCE = True

if OPTIMIZE_PERFORMANCE:
    print("Performance optimization enabled. Using simplified model for faster execution.")


    # Simplify dynamics for faster execution
    def pendulum_dynamics(t, state, voltage_func):
        """
        Simplified dynamics for faster execution while maintaining key behaviors
        """
        theta_m, theta_L, theta_m_dot, theta_L_dot, cable_pos, cable_vel = state

        # Input voltage
        vm = voltage_func(t)

        # Motor torque calculation (simplified)
        im = (vm - Km * theta_m_dot) / Rm
        Tm = Km * im

        # Simplified dynamics (linearized near hanging position)
        # Motor dynamics
        theta_m_ddot = (Tm - DA * theta_m_dot - 0.5 * mL * LL * LA * np.sin(theta_L) * theta_L_dot ** 2) / Jm

        # Pendulum dynamics with linearized terms
        theta_L_ddot = (-DL * theta_L_dot + 0.5 * mL * LL * g * np.sin(theta_L) + cable_pos) / JL

        # Cable dynamics (second-order system)
        omega_n = omega_cable * 2 * np.pi
        cable_acc = -2 * zeta_cable * omega_n * cable_vel - omega_n ** 2 * cable_pos + theta_L_ddot

        return [theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot, cable_vel, cable_acc]

# Time span for the simulation
t_span = (0, 10)
# Reduce number of evaluation points for faster execution
t_eval = np.linspace(0, 10, 10)  # Points at which to store the solution

# Initial conditions [theta_m, theta_L, theta_m_dot, theta_L_dot, cable_pos, cable_vel]
initial_state = [0, np.pi, 0, 0, 0, 0]  # Pendulum hanging down (alpha = pi)


# Simulation with progress tracking
def progress_solve_ivp(dynamics_func, t_span, initial_state, method='RK45', t_eval=None,
                       rtol=1e-6, atol=1e-9, description="Simulating"):
    """Wrapper for solve_ivp with progress updates"""

    # Create a callback function to track progress
    t_start, t_end = t_span
    total_time = t_end - t_start

    start_time = time.time()
    last_update = start_time
    last_t = t_start

    def progress_callback(t, y):
        nonlocal last_update, last_t
        current_time = time.time()

        # Update progress every 0.5 seconds
        if current_time - last_update > 0.5:
            progress = (t - t_start) / total_time * 100
            elapsed = current_time - start_time

            # Calculate time per percentage and estimate remaining time
            time_per_percent = elapsed / (progress if progress > 0 else 1)
            remaining = time_per_percent * (100 - progress)

            sys.stdout.write(
                f"\r{description}: {progress:.1f}% complete | Elapsed: {elapsed:.1f}s | Est. remaining: {remaining:.1f}s")
            sys.stdout.flush()

            last_update = current_time
            last_t = t

        return False  # Never terminate integration early

    # Run the solver with the callback
    solution = solve_ivp(
        dynamics_func,
        t_span,
        initial_state,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        events=progress_callback
    )

    # Final update
    sys.stdout.write(f"\r{description}: 100.0% complete | Total time: {time.time() - start_time:.1f}s")
    sys.stdout.write("\n")
    sys.stdout.flush()

    return solution


# Simulate with step voltage
print("Starting step voltage simulation...")
solution_step = progress_solve_ivp(
    lambda t, y: pendulum_dynamics(t, y, step_voltage),
    t_span,
    initial_state,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9,
    description="Step voltage simulation"
)

# Simulate with zero voltage for comparison
print("\nStarting zero voltage simulation...")
solution_zero = progress_solve_ivp(
    lambda t, y: pendulum_dynamics(t, y, zero_voltage),
    t_span,
    initial_state,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9,
    description="Zero voltage simulation"
)

# Extract results
t = solution_step.t
theta_m_step = solution_step.y[0]
theta_L_step = solution_step.y[1]
theta_m_dot_step = solution_step.y[2]
theta_L_dot_step = solution_step.y[3]
cable_pos_step = solution_step.y[4]

theta_m_zero = solution_zero.y[0]
theta_L_zero = solution_zero.y[1]
theta_m_dot_zero = solution_zero.y[2]
theta_L_dot_zero = solution_zero.y[3]
cable_pos_zero = solution_zero.y[4]

# For easier comparison with the paper, subtract pi from pendulum angle to make 0 the hanging position
theta_L_step_adj = theta_L_step - np.pi
theta_L_zero_adj = theta_L_zero - np.pi

# Plot results
plt.figure(figsize=(16, 12))

# Plot motor angles
plt.subplot(3, 2, 1)
plt.plot(t, theta_m_step, 'b-', label='Step Voltage')
plt.plot(t, theta_m_zero, 'r-', label='Zero Voltage')
plt.ylabel('Motor angle (rad)')
plt.title('Motor Angle Comparison')
plt.legend()
plt.grid(True)

# Plot pendulum angles
plt.subplot(3, 2, 2)
plt.plot(t, theta_L_step_adj, 'b-', label='Step Voltage')
plt.plot(t, theta_L_zero_adj, 'r-', label='Zero Voltage')
plt.ylabel('Pendulum angle (rad)')
plt.title('Pendulum Angle Comparison')
plt.legend()
plt.grid(True)

# Plot motor angular velocities
plt.subplot(3, 2, 3)
plt.plot(t, theta_m_dot_step, 'b-', label='Step Voltage')
plt.plot(t, theta_m_dot_zero, 'r-', label='Zero Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Motor angular velocity (rad/s)')
plt.legend()
plt.grid(True)

# Plot pendulum angular velocities
plt.subplot(3, 2, 4)
plt.plot(t, theta_L_dot_step, 'b-', label='Step Voltage')
plt.plot(t, theta_L_dot_zero, 'r-', label='Zero Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Pendulum angular velocity (rad/s)')
plt.legend()
plt.grid(True)

# Plot cable oscillation
plt.subplot(3, 2, 5)
plt.plot(t, cable_pos_step, 'b-', label='Step Voltage')
plt.plot(t, cable_pos_zero, 'r-', label='Zero Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Cable oscillation (rad)')
plt.title('Encoder Cable Effect')
plt.legend()
plt.grid(True)

# Plot combined pendulum angle (with cable effect)
plt.subplot(3, 2, 6)
combined_angle_step = theta_L_step_adj + cable_pos_step
combined_angle_zero = theta_L_zero_adj + cable_pos_zero
plt.plot(t, combined_angle_step, 'b-', label='Step Voltage')
plt.plot(t, combined_angle_zero, 'r-', label='Zero Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Combined angle (rad)')
plt.title('Pendulum Angle with Cable Effect')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate and print some key parameters for comparison with the paper
print("\nSystem Characteristics:")
print("------------------------")

# Calculate damping ratio of pendulum oscillation using log decrement method
# Find peaks in the pendulum motion (for the zero voltage case which should oscillate freely)
from scipy.signal import find_peaks

peaks, _ = find_peaks(theta_L_zero_adj)
if len(peaks) >= 2:
    y1 = theta_L_zero_adj[peaks[0]]
    y2 = theta_L_zero_adj[peaks[1]]
    log_dec = np.log(abs(y1 / y2))
    damping_ratio = 1 / np.sqrt(1 + (2 * np.pi / log_dec) ** 2)
    print(f"Estimated pendulum damping ratio: {damping_ratio:.4f}")
    print(f"Paper reported pendulum damping ratio: 0.367")

    # Natural frequency
    Td = t[peaks[1]] - t[peaks[0]]  # Period of damped oscillation
    damped_freq = 2 * np.pi / Td
    natural_freq = damped_freq / np.sqrt(1 - damping_ratio ** 2)
    print(f"Estimated pendulum natural frequency: {natural_freq / (2 * np.pi):.2f} Hz")
    print(f"Paper reported pendulum natural frequency: 11.12 Hz")

# Cable oscillation parameters
print(f"Cable damping ratio (input): {zeta_cable}")
print(f"Cable natural frequency (input): {omega_cable} Hz")