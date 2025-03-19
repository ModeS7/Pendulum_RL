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
    def VM(vm):  # Motor voltage dead zone at 0.2V
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


# ============== Energy-based Swing-up Controller ==============
def energy_swing_up(state):
    """
    Energy-based swing-up controller for the pendulum

    The idea is to inject energy into the system when the pendulum is below
    the horizontal, and remove energy when it's above. This pumps the pendulum
    higher with each swing until it reaches the upright position.

    Args:
        state: [theta, alpha, theta_dot, alpha_dot]

    Returns:
        control voltage
    """
    theta, alpha, theta_dot, alpha_dot = state

    # Reference energy for upright position (potential energy at upright)
    # Note: in this system, alpha=π is hanging down, alpha=0 or 2π is upright
    E_ref = Mp * g * Lp  # Energy needed at upright position

    # Current energy of pendulum (kinetic + potential)
    # Potential energy is relative to hanging down position (alpha=π)
    E_kinetic = 0.5 * Jp * alpha_dot ** 2
    E_potential = Mp * g * Lp * (np.cos(alpha) - (-1))  # -1 is cos(π)
    E_total = E_kinetic + E_potential

    # Energy error
    E_error = E_ref - E_total

    # Swing-up control law
    k_energy = 0.8  # Increased energy control gain for more aggressive swing-up

    # For small oscillations at the beginning, give a stronger push
    if abs(alpha - np.pi) < 0.1 and abs(alpha_dot) < 0.2:
        # Initial push to get the pendulum moving
        return 5.0 * np.sign(np.sin(theta))

    # Use the sign of pendulum velocity and sign of pendulum position
    # to determine when to apply torque
    control = k_energy * E_error * np.sign(alpha_dot * np.sin(alpha))

    # Add direct term to ensure arm is moving
    control += 0.5 * np.sign(np.sin(alpha)) - 0.1 * theta_dot  # Add damping

    # Limit control output
    max_voltage = 5.0
    control = np.clip(control, -max_voltage, max_voltage)

    return control


# ============== PID Controller for Stabilization ==============
class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Target angle (0 = upright)
        self.integral = 0  # Integral term
        self.prev_error = 0  # Previous error for derivative
        self.prev_time = 0  # Previous time for dt calculation

    def compute(self, t, state):
        """
        Compute PID control action

        Args:
            t: current time
            state: [theta, alpha, theta_dot, alpha_dot]

        Returns:
            control voltage
        """
        theta, alpha, theta_dot, alpha_dot = state

        # Calculate dt (time step)
        dt = t - self.prev_time if t > self.prev_time else 0.001
        self.prev_time = t

        # Calculate error (relative to upright position)
        # Note: We want alpha = 0 (or 2π), and current reference is α=π for hanging down
        error = (alpha - np.pi) - self.setpoint  # Get angle relative to upright

        # Normalize angle to [-π, π]
        error = ((error + np.pi) % (2 * np.pi)) - np.pi

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10.0, 10.0)  # Anti-windup
        i_term = self.ki * self.integral

        # Derivative term (use actual measured alpha_dot instead of error derivative for cleaner signal)
        d_term = self.kd * alpha_dot  # Use pendulum angular velocity directly

        # Total control effort
        control = -(p_term + i_term + d_term)

        # Add arm position control to keep the arm near center
        control -= 1.0 * theta + 0.5 * theta_dot

        # Limit control output
        max_voltage = 5.0
        control = np.clip(control, -max_voltage, max_voltage)

        # Update previous error for next iteration
        self.prev_error = error

        return control


# ============== Combined Controller (Swing-up + PID) ==============
class CombinedController:
    def __init__(self):
        # PID gains tuned for stabilization - increased for better performance
        self.pid = PIDController(kp=20.0, ki=1.0, kd=2.0)
        self.control_history = []  # For storing control voltages
        self.state_history = []  # For storing system states
        self.swing_time = 0  # Time in swing-up mode
        self.stabilize_time = 0  # Time in stabilization mode
        self.control_mode = "swing-up"  # Track current control mode

    def compute(self, t, state):
        """
        Combined controller that switches between swing-up and stabilization

        Args:
            t: current time
            state: [theta, alpha, theta_dot, alpha_dot]

        Returns:
            control voltage
        """
        theta, alpha, theta_dot, alpha_dot = state

        # Store state for analysis
        self.state_history.append((t, state))

        # Normalize pendulum angle relative to upright (0 = upright, ±π = hanging down)
        alpha_norm = ((alpha - np.pi + np.pi) % (2 * np.pi)) - np.pi
        alpha_norm = alpha

        # Define region where PID controller takes over
        # When pendulum is within ±0.4 radians of upright position and not moving too fast
        stabilization_region = 0.4  # radians

        # Determine control mode
        if abs(alpha_norm) < stabilization_region and abs(alpha_dot) < 2.0:
            # Use PID control for stabilization
            control = self.pid.compute(t, state)
            if self.control_mode != "stabilize":
                print(f"Switching to stabilization at t={t:.2f}s")
                self.control_mode = "stabilize"
            self.stabilize_time += 1
        else:
            # Use energy-based swing-up
            control = energy_swing_up(state)
            if self.control_mode != "swing-up":
                print(f"Switching to swing-up at t={t:.2f}s")
                self.control_mode = "swing-up"
            self.swing_time += 1

        # Store control for plotting
        self.control_history.append((t, control))

        return control


# ============== Simulation of Combined Controller ==============
def run_controlled_simulation():
    # Create combined controller
    controller = CombinedController()

    # Define voltage function that uses the controller
    def controlled_voltage(t):
        # This is called by the ODE solver, which doesn't pass state
        # We need to get the state from the solution at previous time steps

        # Initial state (if t=0)
        if t == 0 or len(t_points) == 0:
            return 0.0

        # Find closest time point
        idx = np.argmin(np.abs(np.array(t_points) - t))
        if idx < len(states):
            return controller.compute(t, states[idx])
        return 0.0

    # Storage for states during simulation
    t_points = []
    states = []

    # Event function to record states during simulation
    def record_state(t, y):
        t_points.append(t)
        states.append(y.copy())
        return False  # Continue integration

    # Time span for the simulation
    t_span = (0, 20)  # 20 seconds
    t_eval = np.linspace(0, 20, 10000)

    # Initial conditions (pendulum hanging down)
    initial_state = [0, 2, 0, 0]  # [theta, alpha, theta_dot, alpha_dot]

    # Solve using solve_ivp with the controlled voltage
    print("Simulating with swing-up and PID control...")
    solution = solve_ivp(
        lambda t, y: lagrangian_dynamics(t, y, controlled_voltage),
        t_span,
        initial_state,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        events=record_state
    )

    # Extract control voltages for plotting
    control_times, control_values = zip(*controller.control_history) if controller.control_history else ([], [])

    return solution, np.array(control_times), np.array(control_values)


# ============== Alternative Implementation Using RK4 Integration ==============
# This approach gives more direct control over the simulation process
def run_controlled_simulation_rk4():
    # Create combined controller
    controller = CombinedController()

    # Time settings
    t_end = 15.0
    dt = 0.001
    num_steps = int(t_end / dt)
    t = np.linspace(0, t_end, num_steps)

    # Arrays to store results
    theta = np.zeros(num_steps)
    alpha = np.zeros(num_steps)
    theta_dot = np.zeros(num_steps)
    alpha_dot = np.zeros(num_steps)
    control_voltages = np.zeros(num_steps)

    # Initial state (pendulum hanging down with small perturbation to break symmetry)
    state = np.array([0, np.pi + 0.01, 0, 0])  # [theta, alpha, theta_dot, alpha_dot]
    theta[0], alpha[0], theta_dot[0], alpha_dot[0] = state

    # RK4 integration step function
    def rk4_step(func, t, y, h, voltage):
        # Define a voltage function that returns the constant voltage
        def voltage_func(_):
            return voltage

        # Calculate derivatives
        k1 = np.array(func(t, y, voltage_func))
        k2 = np.array(func(t + 0.5 * h, y + 0.5 * h * k1, voltage_func))
        k3 = np.array(func(t + 0.5 * h, y + 0.5 * h * k2, voltage_func))
        k4 = np.array(func(t + h, y + h * k3, voltage_func))

        # Update state
        return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Add some initial energy (small push)
    control_voltages[0] = 3.0

    # Simulation loop
    for i in range(1, num_steps):
        # Calculate control voltage
        control = controller.compute(t[i - 1], state)
        control_voltages[i - 1] = control

        # RK4 integration step
        state = rk4_step(lagrangian_dynamics, t[i - 1], state, dt, control)

        # Store state
        theta[i], alpha[i], theta_dot[i], alpha_dot[i] = state

        # Print progress at intervals
        if i % 1000 == 0:
            print(f"Time: {t[i]:.2f}s, Pendulum angle: {(alpha[i] - np.pi):.4f}, Arm angle: {theta[i]:.4f}")

    # Store the last control voltage
    control_voltages[-1] = controller.compute(t[-1], state)

    # Print summary
    print(
        f"Simulation complete. Swing-up time: {controller.swing_time * dt:.2f}s, Stabilization time: {controller.stabilize_time * dt:.2f}s")

    # Create a solution object similar to solve_ivp output
    class Solution:
        def __init__(self):
            self.t = t
            self.y = np.array([theta, alpha, theta_dot, alpha_dot])

    return Solution(), t, control_voltages


# ============== Run and Visualize the Simulation ==============
def main():
    # Use RK4 method for more control over the simulation
    solution, t_control, control_voltages = run_controlled_simulation_rk4()

    # Extract results
    t = solution.t
    theta = solution.y[0]
    alpha = solution.y[1]
    theta_dot = solution.y[2]
    alpha_dot = solution.y[3]

    # Calculate energy over time (for analysis)
    energy = np.zeros_like(t)
    for i in range(len(t)):
        E_kinetic = 0.5 * Jp * alpha_dot[i] ** 2
        E_potential = Mp * g * Lp * (np.cos(alpha[i]) - (-1))
        energy[i] = E_kinetic + E_potential

    # Unwrap pendulum angle for better visualization
    alpha_unwrapped = np.unwrap(alpha)

    # Create 3x3 grid of plots for comprehensive analysis
    plt.figure(figsize=(18, 12))

    # Plot arm angle
    plt.subplot(3, 3, 1)
    plt.plot(t, theta, 'b-')
    plt.ylabel('Arm angle (rad)')
    plt.title('Arm Angle (θ)')
    plt.grid(True)

    # Plot pendulum angle
    plt.subplot(3, 3, 2)
    plt.plot(t, alpha, 'r-')
    #plt.axhline(y=0, color='k', linestyle='--')  # Upright position
    #plt.axhline(y=np.pi, color='g', linestyle='--')  # Hanging position
    #plt.axhline(y=-np.pi, color='g', linestyle='--')  # Hanging position
    plt.ylabel('Pendulum angle (rad)')
    plt.title('Pendulum Angle Relative to Upright')
    plt.grid(True)

    # Plot arm angular velocity
    plt.subplot(3, 3, 4)
    plt.plot(t, theta_dot, 'b-')
    plt.ylabel('Arm velocity (rad/s)')
    plt.title('Arm Angular Velocity')
    plt.grid(True)

    # Plot pendulum angular velocity
    plt.subplot(3, 3, 5)
    plt.plot(t, alpha_dot, 'r-')
    plt.ylabel('Pendulum velocity (rad/s)')
    plt.title('Pendulum Angular Velocity')
    plt.grid(True)

    # Plot control voltage
    plt.subplot(3, 3, 7)
    plt.plot(t_control, control_voltages, 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Control voltage (V)')
    plt.title('Control Voltage')
    plt.grid(True)

    # Plot pendulum phase portrait
    plt.subplot(3, 3, 8)
    plt.plot(alpha_unwrapped - np.pi, alpha_dot, 'r-')
    plt.xlabel('Pendulum angle (rad)')
    plt.ylabel('Pendulum velocity (rad/s)')
    plt.title('Pendulum Phase Portrait')
    plt.grid(True)

    # Plot total energy of the pendulum
    plt.subplot(3, 3, 3)
    plt.plot(t, energy, 'purple')
    plt.ylabel('Energy (J)')
    plt.title('Pendulum Energy')
    plt.grid(True)

    # Plot pendulum vs arm motion (2D projection)
    plt.subplot(3, 3, 6)
    plt.plot(theta, alpha_unwrapped - np.pi, 'b-')
    plt.xlabel('Arm angle (rad)')
    plt.ylabel('Pendulum angle (rad)')
    plt.title('2D System Trajectory')
    plt.grid(True)

    # Plot animation frames for visualization (select points)
    plt.subplot(3, 3, 9)
    num_frames = 10
    frame_indices = np.linspace(0, len(t) - 1, num_frames).astype(int)

    # Plot arm and pendulum positions
    L_arm = Lr  # Arm length
    L_pend = Lp  # Pendulum length

    for i in frame_indices:
        # Arm endpoint
        arm_x = L_arm * np.sin(theta[i])
        arm_y = -L_arm * np.cos(theta[i])

        # Pendulum endpoint
        pend_x = arm_x + L_pend * np.sin(theta[i] + alpha[i] - np.pi)
        pend_y = arm_y - L_pend * np.cos(theta[i] + alpha[i] - np.pi)

        # Plot arm
        plt.plot([0, arm_x], [0, arm_y], 'b-', linewidth=1)

        # Plot pendulum
        plt.plot([arm_x, pend_x], [arm_y, pend_y], 'r-', linewidth=1)

        # Annotate time
        plt.annotate(f"{t[i]:.1f}s", (arm_x, arm_y), fontsize=8)

    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('System Animation Frames')
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # Create a separate animation figure
    plt.figure(figsize=(10, 8))
    plt.title("Inverted Pendulum Motion")

    # More dense sampling for smoother animation
    frame_indices = np.linspace(0, len(t) - 1, 50).astype(int)

    """for i in frame_indices:
        plt.clf()  # Clear figure

        # Arm endpoint
        arm_x = L_arm * np.sin(theta[i])
        arm_y = -L_arm * np.cos(theta[i])

        # Pendulum endpoint
        pend_x = arm_x + L_pend * np.sin(theta[i] + alpha[i] - np.pi)
        pend_y = arm_y - L_pend * np.cos(theta[i] + alpha[i] - np.pi)

        # Plot pivot point
        plt.plot(0, 0, 'ko', markersize=8)

        # Plot arm
        plt.plot([0, arm_x], [0, arm_y], 'b-', linewidth=3)
        plt.plot(arm_x, arm_y, 'bo', markersize=6)

        # Plot pendulum
        plt.plot([arm_x, pend_x], [arm_y, pend_y], 'r-', linewidth=3)
        plt.plot(pend_x, pend_y, 'ro', markersize=6)

        # Add information
        plt.annotate(f"Time: {t[i]:.2f}s", (-0.25, 0.15), fontsize=12)
        plt.annotate(f"Pendulum angle: {(alpha[i] - np.pi):.2f} rad", (-0.25, 0.1), fontsize=12)
        plt.annotate(f"Arm angle: {theta[i]:.2f} rad", (-0.25, 0.05), fontsize=12)
        plt.annotate(f"Control: {control_voltages[i]:.2f} V", (-0.25, 0), fontsize=12)

        plt.xlim(-0.3, 0.3)
        plt.ylim(-0.3, 0.2)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid(True)
        plt.axis('equal')

        plt.pause(0.05)  # Pause to create animation effect

    plt.show()"""


if __name__ == "__main__":
    main()