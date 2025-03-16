import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System Parameters (from the QUBE-Servo 2 manual)
# Motor and Pendulum parameters
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
Lp = 0.129  # Pendulum length from pivot to center of mass (m) (0.085 + 0.129)/2
Jp = (1/3) * Mp * Lp ** 2   # Pendulum moment of inertia (kg·m²)
Br = 0.001  # Rotary arm viscous damping coefficient (N·m·s/rad)
Bp = 0.0001  # Pendulum viscous damping coefficient (N·m·s/rad)
g = 9.81  # Gravity constant (m/s²)
Jr = Jm + Jh + Mr * Lr ** 2 / 3  # Assuming arm is like a rod pivoting at one end

# Simulation parameters
dt = 0.001  # Time step size (s)
t_end = 10  # Simulation duration (s)
t = np.arange(0, t_end, dt)
num_points = len(t)


# Function to compute derivatives for the pendulum system
def pendulum_dynamics(state, vm):
    """
    Compute derivatives for the QUBE-Servo 2 pendulum system

    state = [theta, alpha, theta_dot, alpha_dot]
    where:
        theta = rotary arm angle
        alpha = pendulum angle
        theta_dot = rotary arm angular velocity
        alpha_dot = pendulum angular velocity

    Returns [theta_dot, alpha_dot, theta_ddot, alpha_ddot]
    """
    theta, alpha, theta_dot, alpha_dot = state

    # Motor current
    im = (vm - km * theta_dot) / Rm

    # Motor torque
    tau = kt * im

    # Equations of motion
    # Inertia matrix elements
    M11 = Jr + Mp * Lr ** 2
    M12 = Mp * Lr * Lp / 2 * np.cos(alpha)
    M21 = M12
    M22 = Jp

    # Coriolis and centrifugal terms
    C1 = -Mp * Lr * (Lp / 2) * alpha_dot ** 2 * np.sin(alpha) - Br * theta_dot
    C2 = Mp * g * (Lp / 2) * np.sin(alpha) - Bp * alpha_dot

    # Torque input vector
    B1 = tau
    B2 = 0

    # Solve for the accelerations
    det_M = M11 * M22 - M12 * M21

    # Check for singularity
    if abs(det_M) < 1e-10:
        det_M = np.sign(det_M) * 1e-10

    theta_ddot = (M22 * (B1 + C1) - M12 * (B2 + C2)) / det_M
    alpha_ddot = (M11 * (B2 + C2) - M21 * (B1 + C1)) / det_M

    return np.array([theta_dot, alpha_dot, theta_ddot, alpha_ddot])

# Animate the pendulum motion
def animate_pendulum(t, theta, alpha, dt_anim=0.05):
    """
    Animate the pendulum motion

    Parameters:
    t: time array
    theta: rotary arm angle array
    alpha: pendulum angle array
    dt_anim: animation time step (s)
    """
    from matplotlib import animation

    # Convert animation time step to indices
    step = int(dt_anim / dt)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-0.2, 0.2), ylim=(-0.2, 0.2))
    ax.grid()

    arm_length = 0.085  # Rotary arm length (m)
    pend_length = 0.129  # Full pendulum length (m)

    # Elements to be animated
    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        i = i * step
        if i >= len(t):
            i = len(t) - 1

        # Rotary arm endpoint
        arm_x = arm_length * np.cos(theta[i])
        arm_y = arm_length * np.sin(theta[i])

        # Pendulum endpoint
        pend_x = arm_x + pend_length * np.sin(alpha[i])
        pend_y = arm_y - pend_length * np.cos(alpha[i])

        # Update line data
        line.set_data([0, arm_x, pend_x], [0, arm_y, pend_y])
        time_text.set_text(time_template % t[i])
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=len(t) // step,
                                  interval=dt_anim * 1000, blit=True, init_func=init)

    plt.title('QUBE-Servo 2 Pendulum Animation')
    plt.show()

    return ani


# Initialize state variables [theta, alpha, theta_dot, alpha_dot]
# Starting with pendulum in downward position (alpha = π)
state = np.zeros((num_points, 4))
state[0] = [0, np.pi, 0, 0]  # Initial conditions

# Input voltage (you can modify this)
# Example: Step input
vm = np.zeros(num_points)
vm[int(1 / dt):] = 3.0  # 3V step input after 1 second

# Euler Integration
for i in range(1, num_points):
    derivatives = pendulum_dynamics(state[i - 1], vm[i - 1])
    state[i] = state[i - 1] + dt * derivatives

# Unwrap pendulum angle for continuous plotting
alpha_unwrapped = np.unwrap(state[:, 1])

# Call the animate_pendulum function to visualize the motion
ani = animate_pendulum(t, state[:, 0], alpha_unwrapped)

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t, vm)
plt.ylabel('Input Voltage (V)')
plt.title('QUBE-Servo 2 Pendulum Simulation')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, state[:, 0], label='Arm angle (θ)')
plt.plot(t, alpha_unwrapped - np.pi, label='Pendulum angle (α)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, state[:, 2], label='Arm angular velocity')
plt.plot(t, state[:, 3], label='Pendulum angular velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# To animate the pendulum motion, uncomment the following line
# ani = animate_pendulum(t, state[:, 0], state[:, 1])

# Function to implement control for pendulum swing-up and balancing
def pendulum_control_simulation(controller_type='energy'):
    """
    Simulate the pendulum with a controller

    controller_type: 'energy' for energy-based swing-up, 'lqr' for LQR balance
    """
    # Initialize state variables [theta, alpha, theta_dot, alpha_dot]
    state = np.zeros((num_points, 4))
    state[0] = [0, np.pi, 0, 0]  # Initial conditions

    # Control input
    vm = np.zeros(num_points)

    # Energy-based swing-up controller parameters
    k_energy = 0.1
    E_ref = Mp * g * Lp * 2  # Reference energy for upright position

    # LQR parameters (example values - would need proper design)
    K_lqr = np.array([2.0, 15.0, 1.0, 1.5])  # LQR gain

    # Reference for upright position
    alpha_ref = 0  # Upright position

    for i in range(1, num_points):
        # Current state
        theta, alpha, theta_dot, alpha_dot = state[i - 1]

        # Determine control based on controller type
        if controller_type == 'energy':
            # Energy-based swing-up
            # Calculate system energy (kinetic + potential)
            E_kin = 0.5 * Jp * alpha_dot ** 2
            E_pot = Mp * g * Lp * (1 - np.cos(alpha - np.pi))
            E_total = E_kin + E_pot

            # Control law (energy shaping)
            vm[i] = k_energy * alpha_dot * np.cos(alpha) * (E_total - E_ref)

            # Saturate control input
            vm[i] = np.clip(vm[i], -3.0, 3.0)

        elif controller_type == 'lqr':
            # LQR balance controller
            # Normalize pendulum angle to reference
            alpha_norm = (alpha - np.pi) % (2 * np.pi)
            if alpha_norm > np.pi:
                alpha_norm -= 2 * np.pi

            # Control law
            vm[i] = -np.dot(K_lqr, [theta, alpha_norm, theta_dot, alpha_dot])

            # Saturate control input
            vm[i] = np.clip(vm[i], -3.0, 3.0)

        else:  # Simple PD controller for testing
            # PD control for pendulum angle
            kp = 0.5
            kd = 0.1
            alpha_error = (alpha - alpha_ref + np.pi) % (2 * np.pi) - np.pi
            vm[i] = -kp * alpha_error - kd * alpha_dot
            vm[i] = np.clip(vm[i], -3.0, 3.0)

        # Euler integration
        derivatives = pendulum_dynamics(state[i - 1], vm[i])
        state[i] = state[i - 1] + dt * derivatives

    return t, state, vm

# To run control simulation, uncomment one of these lines
# t, controlled_state, control_vm = pendulum_control_simulation('energy')
# t, controlled_state, control_vm = pendulum_control_simulation('lqr')