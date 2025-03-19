import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import sys
from scipy.signal import find_peaks

#   System Dynamics and Parameters from the system identification document:
#   System identiﬁcations of a 2DOF pendulum controlled
#   by QUBE-servo and its unwanted oscillation factors
#   https://bibliotekanauki.pl/articles/1845011.pdf

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


# Optimized dynamics function without cable effect
def pendulum_dynamics_no_cable(t, state, voltage_func):
    """
    Dynamics function for the 2DOF QUBE-Servo pendulum without cable effect

    state = [theta_m, theta_L, theta_m_dot, theta_L_dot]
    where:
        theta_m = motor angle (pendulum arm angle)
        theta_L = pendulum link angle
        theta_m_dot, theta_L_dot = respective angular velocities
    """
    theta_m, theta_L, theta_m_dot, theta_L_dot = state

    # Input voltage
    vm = voltage_func(t)

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

    return [theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot]


# Voltage functions
def step_voltage(t):
    if t <= 0.2:
        return 0.0
    elif t < 10.0:
        return 3.0
    else:
        return 0.0

def zero_voltage(t):
    if t <= 0.2:
        return 0.0
    elif t < 10.0:
        return -3.0
    else:
        return 0.0


# Use a shorter time span for faster execution
t_span = (0, 15.0)
t_eval = np.linspace(0, 15.0, 1500)  # More points for smoother plots

# Initial conditions [theta_m, theta_L, theta_m_dot, theta_L_dot]
initial_state = [0, np.pi, 0, 0]  # Pendulum hanging down (alpha = pi)

# Run simulations
print("Starting step voltage simulation...")
start_time = time.time()
solution_step = solve_ivp(
    lambda t, y: pendulum_dynamics_no_cable(t, y, step_voltage),
    t_span,
    initial_state,
    method='DOP853',
    t_eval=t_eval,
    rtol=1e-3,
    atol=1e-6
)
step_time = time.time() - start_time
print(f"Step voltage simulation completed in {step_time:.2f} seconds")

print("\nStarting zero voltage simulation...")
start_time = time.time()
solution_zero = solve_ivp(
    lambda t, y: pendulum_dynamics_no_cable(t, y, zero_voltage),
    t_span,
    initial_state,
    method='DOP853',
    t_eval=t_eval,
    rtol=1e-3,
    atol=1e-6
)
zero_time = time.time() - start_time
print(f"Zero voltage simulation completed in {zero_time:.2f} seconds")

# Extract results
t = solution_step.t
theta_m_step = solution_step.y[0]
theta_L_step = solution_step.y[1]
theta_m_dot_step = solution_step.y[2]
theta_L_dot_step = solution_step.y[3]

theta_m_zero = solution_zero.y[0]
theta_L_zero = solution_zero.y[1]
theta_m_dot_zero = solution_zero.y[2]
theta_L_dot_zero = solution_zero.y[3]

# For easier comparison with the paper, subtract pi from pendulum angle
theta_L_step_adj = theta_L_step - np.pi
theta_L_zero_adj = theta_L_zero - np.pi

# Plot results
plt.figure(figsize=(14, 10))

# Plot motor angles
plt.subplot(2, 2, 1)
plt.plot(t, theta_m_step, 'b-', label='Step Voltage')
plt.plot(t, theta_m_zero, 'r-', label='Zero Voltage')
plt.ylabel('Motor angle (rad)')
plt.title('Motor Angle Comparison (No Cable Effect)')
plt.legend()
plt.grid(True)

# Plot pendulum angles
plt.subplot(2, 2, 2)
plt.plot(t, theta_L_step_adj, 'b-', label='Step Voltage')
plt.plot(t, theta_L_zero_adj, 'r-', label='Zero Voltage')
plt.ylabel('Pendulum angle (rad)')
plt.title('Pendulum Angle Comparison (No Cable Effect)')
plt.legend()
plt.grid(True)

# Plot motor angular velocities
plt.subplot(2, 2, 3)
plt.plot(t, theta_m_dot_step, 'b-', label='Step Voltage')
plt.plot(t, theta_m_dot_zero, 'r-', label='Zero Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Motor angular velocity (rad/s)')
plt.legend()
plt.grid(True)

# Plot pendulum angular velocities
plt.subplot(2, 2, 4)
plt.plot(t, theta_L_dot_step, 'b-', label='Step Voltage')
plt.plot(t, theta_L_dot_zero, 'r-', label='Zero Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Pendulum angular velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('qube_servo_no_cable_results.png')
plt.show()

# Calculate and print some key parameters
print("\nSystem Characteristics (No Cable Effect):")
print("----------------------------------------")

# Calculate damping ratio of pendulum oscillation using log decrement method
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