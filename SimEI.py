import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System Parameters (from the QUBE-Servo 2 manual)
Rm = 8.4  # Motor resistance (Ohm)
kt = 0.042  # Motor torque constant (N·m/A)
km = 0.042  # Motor back-EMF constant (V·s/rad)
Jm = 4e-6  # Motor moment of inertia (kg·m²)
mh = 0.016  # Hub mass (kg)
rh = 0.0111  # Hub radius (m)
Jh = 0.6e-6  # Hub moment of inertia (kg·m^2)
Mr = 0.095  # Rotary arm mass (kg)
Lr = 0.085  # Arm length, pivot to end (m)
Mp = 0.024  # Pendulum mass (kg)
Lp = 0.129  # Pendulum length from pivot to center of mass (m)
Jp = (1 / 3) * Mp * Lp ** 2  # Pendulum moment of inertia (kg·m²)
Br = 0.001  # Rotary arm viscous damping coefficient (N·m·s/rad)
Bp = 0.0001  # Pendulum viscous damping coefficient (N·m·s/rad)
g = 9.81  # Gravity constant (m/s²)
Jr = Jm + Jh + Mr * Lr ** 2 / 3  # Total rotary arm inertia

def lagrangian_dynamics(t, state, voltage_func):
    """
    Lagrangian formulation of the QUBE-Servo 2 dynamics

    state = [theta, alpha, theta_dot, alpha_dot]
    where:
        theta = rotary arm angle
        alpha = pendulum angle (α=π is hanging down)
        theta_dot, alpha_dot = respective angular velocities
    """
    theta, alpha, theta_dot, alpha_dot = state

    # Input voltage
    def VM(vm):     #  Motor voltage dead zone at 0.2V
        return 0 if -0.2 <= vm <= 0.2 else vm
    vm = VM(voltage_func(t))
    # Motor current
    im = (vm - km * theta_dot) / Rm

    # Motor torque
    tau = kt * im

    # Inertia matrix elements
    M11 = Jr + Mp * Lr ** 2
    M12 = Mp * Lr * Lp / 2 * np.cos(alpha)
    M21 = M12
    M22 = Jp
    det_M = M11 * M22 - M12 * M21

    # Nonlinear terms
    C = -Mp * Lr * Lp * np.sin(alpha) * alpha_dot ** 2  # For arm
    G = -Mp * Lp * g * np.sin(alpha)  # Gravity for pendulum
    F_theta = Br * theta_dot
    F_alpha = Bp * alpha_dot

    # Solve for accelerations: M * [theta_ddot; alpha_ddot] = [tau - C - F_theta; -G - F_alpha]
    M = np.array([[M11, M12], [M21, M22]])
    # Right-hand side with centrifugal term added for pendulum
    rhs = np.array([
        tau - C - F_theta,
        -G - F_alpha - Mp * Lr * Lp * np.sin(alpha) * theta_dot ** 2
    ])
    acc = np.linalg.solve(M, rhs)

    # Solve for accelerations
    if abs(det_M) < 1e-10:  # Handle near-singular matrix
        theta_ddot = 0
        alpha_ddot = 0
    else:
        theta_ddot = acc[0]
        alpha_ddot = acc[1]

    return [theta_dot, alpha_dot, theta_ddot, alpha_ddot]

# Voltage functions
def positive_voltage(t):
    if 10.0 >= t >= 0.2:
        return 3.0
    else:
        return -0.0
def negative_voltage(t):
    if 10.0 >= t >= 0.2:
        return 0.0
    else:
        return 0.0

# Time span for the simulation
t_span = (0, 15)
t_eval = np.linspace(0, 15, 15000)  # Points at which to store the solution

# Initial conditions
initial_state = [0, np.pi, 0, 0]  # [theta, alpha, theta_dot, alpha_dot]

# Solve using solve_ivp
print("Simulating positive voltage...")
solution_pos = solve_ivp(
    lambda t, y: lagrangian_dynamics(t, y, positive_voltage),
    t_span,
    initial_state,
    method='RK45',  # Runge-Kutta 4(5)
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9  # Tight tolerances for better accuracy
)

print("Simulating negative voltage...")
solution_neg = solve_ivp(
    lambda t, y: lagrangian_dynamics(t, y, negative_voltage),
    t_span,
    initial_state,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9
)

# Extract results for plotting
t = solution_pos.t
theta_pos = solution_pos.y[0]
alpha_pos = solution_pos.y[1]
theta_dot_pos = solution_pos.y[2]
alpha_dot_pos = solution_pos.y[3]

theta_neg = solution_neg.y[0]
alpha_neg = solution_neg.y[1]
theta_dot_neg = solution_neg.y[2]
alpha_dot_neg = solution_neg.y[3]

# Unwrap angles for continuous plotting
alpha_pos_unwrapped = np.unwrap(alpha_pos)
alpha_neg_unwrapped = np.unwrap(alpha_neg)

# Plot results
plt.figure(figsize=(14, 10))

# Plot arm angles
plt.subplot(2, 2, 1)
plt.plot(t, theta_pos, 'b-', label='+3.0V')
plt.plot(t, theta_neg, 'r-', label='-3.0V')
plt.ylabel('Arm angle (rad)')
plt.title('Arm Angle Comparison (Lagrangian Model)')
plt.legend()
plt.grid(True)

# Plot pendulum angles
plt.subplot(2, 2, 2)
plt.plot(t, alpha_pos_unwrapped - np.pi, 'b-', label='+3.0V')
plt.plot(t, alpha_neg_unwrapped - np.pi, 'r-', label='-3.0V')
#plt.ylim(-np.pi, np.pi)
plt.ylabel('Pendulum angle (rad)')
plt.title('Pendulum Angle Comparison')
plt.legend()
plt.grid(True)

# Plot arm angular velocities
plt.subplot(2, 2, 3)
plt.plot(t, theta_dot_pos, 'b-', label='+3.0V')
plt.plot(t, theta_dot_neg, 'r-', label='-3.0V')
plt.xlabel('Time (s)')
plt.ylabel('Arm angular velocity (rad/s)')
plt.legend()
plt.grid(True)

# Plot pendulum angular velocities
plt.subplot(2, 2, 4)
plt.plot(t, alpha_dot_pos, 'b-', label='+3.0V')
plt.plot(t, alpha_dot_neg, 'r-', label='-3.0V')
plt.xlabel('Time (s)')
plt.ylabel('Pendulum angular velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()