import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb
from time import time
from dataclasses import dataclass
from collections import deque
import argparse

# System Constants
MAX_VOLTAGE = 10.0
THETA_MIN = -2.2
THETA_MAX = 2.2


# ============= Utility Functions =============
@nb.njit(fastmath=True, cache=True)
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


@nb.njit(fastmath=True, cache=True)
def enforce_theta_limits(state, theta_min=THETA_MIN, theta_max=THETA_MAX):
    """Enforce hard limits on theta angle and velocity"""
    theta, alpha, theta_dot, alpha_dot = state

    # Apply hard limit on theta
    if theta > theta_max:
        theta = theta_max
        # If hitting upper limit with positive velocity, stop the motion
        if theta_dot > 0:
            theta_dot = 0.0
    elif theta < theta_min:
        theta = theta_min
        # If hitting lower limit with negative velocity, stop the motion
        if theta_dot < 0:
            theta_dot = 0.0

    return np.array([theta, alpha, theta_dot, alpha_dot])


# ============= System Parameters =============
@dataclass
class SystemParameters:
    """Dataclass to store system parameters with randomization capabilities"""
    # Base values
    Rm: float = 8.94  # Motor resistance (Ohm)
    Km: float = 0.0431  # Motor back-emf constant
    Jm: float = 6e-5  # Total moment of inertia acting on motor shaft (kg·m^2)
    bm: float = 3e-4  # Viscous damping coefficient (Nm/rad/s)
    DA: float = 3e-4  # Damping coefficient of pendulum arm (Nm/rad/s)
    DL: float = 5e-4  # Damping coefficient of pendulum link (Nm/rad/s)
    mA: float = 0.053  # Weight of pendulum arm (kg)
    mL: float = 0.024  # Weight of pendulum link (kg)
    LA: float = 0.086  # Length of pendulum arm (m)
    LL: float = 0.128  # Length of pendulum link (m)
    JA: float = 5.72e-5  # Inertia moment of pendulum arm (kg·m^2)
    JL: float = 1.31e-4  # Inertia moment of pendulum link (kg·m^2)
    g: float = 9.81  # Gravity constant (m/s^2)
    dt: float = 0.0115  # Base timestep (matches real hardware)
    dt_current: float = 0.0115  # Current timestep for this step

    # Derived parameters (computed when randomized)
    half_mL_LL_g: float = 0.0
    half_mL_LL_LA: float = 0.0
    quarter_mL_LL_squared: float = 0.0
    Mp_g_Lp: float = 0.0
    Jp: float = 0.0

    def randomize(self, randomization_factor=0.1, dt_randomization_factor=0.0):
        """Randomize physical parameters and timestep"""
        # Reset current dt
        self.dt_current = self.dt

        if randomization_factor <= 0.0 and dt_randomization_factor <= 0.0:
            # Skip randomization if factors are zero
            self.update_derived_parameters()
            return self

        # Function to randomize a parameter within the given factor
        def rand_param(value, factor):
            if factor <= 0:
                return value
            return value * (1.0 + np.random.uniform(-factor, factor))

        # Randomize physical parameters
        if randomization_factor > 0:
            self.Rm = rand_param(8.94, randomization_factor)
            self.Km = rand_param(0.0431, randomization_factor)
            self.Jm = rand_param(6e-5, randomization_factor)
            self.bm = rand_param(3e-4, randomization_factor)
            self.DA = rand_param(3e-4, randomization_factor)
            self.DL = rand_param(5e-4, randomization_factor)
            self.mA = rand_param(0.053, randomization_factor)
            self.mL = rand_param(0.024, randomization_factor)
            self.LA = rand_param(0.086, randomization_factor)
            self.LL = rand_param(0.128, randomization_factor)
            self.JA = rand_param(5.72e-5, randomization_factor)
            self.JL = rand_param(1.31e-4, randomization_factor)
            self.g = rand_param(9.81, randomization_factor)  # Simulates different mounting angles

        # Randomize timestep if requested
        if dt_randomization_factor > 0:
            self.dt_current = rand_param(self.dt, dt_randomization_factor)

        # Update derived parameters
        self.update_derived_parameters()
        return self

    def update_derived_parameters(self):
        """Update the derived parameters based on current physical parameters"""
        self.half_mL_LL_g = 0.5 * self.mL * self.LL * self.g
        self.half_mL_LL_LA = 0.5 * self.mL * self.LL * self.LA
        self.quarter_mL_LL_squared = 0.25 * self.mL * self.LL ** 2
        self.Mp_g_Lp = self.mL * self.g * self.LL
        self.Jp = (1 / 3) * self.mL * self.LL ** 2  # Pendulum moment of inertia


# ============= Pendulum Environment =============
class PendulumEnv:
    def __init__(self, dt=0.0115, max_steps=1000, delay_steps=5, delay_range=None,
                 randomization_factor=0.1, dt_randomization_factor=0.0):
        # Initialize parameters
        self.max_steps = max_steps
        self.step_count = 0
        self.state = None
        self.elapsed_time = 0.0
        self.base_dt = dt
        self.dt_randomization_factor = dt_randomization_factor
        self.dt_history = []
        self.randomization_factor = randomization_factor
        self.max_voltage = MAX_VOLTAGE
        self.THETA_MIN = THETA_MIN
        self.THETA_MAX = THETA_MAX

        # Setup delay mechanism
        self.min_delay = delay_steps
        self.max_delay = delay_steps
        if delay_range is not None and isinstance(delay_range, (list, tuple)) and len(delay_range) == 2:
            self.min_delay = max(0, delay_range[0])
            self.max_delay = max(self.min_delay, delay_range[1])

        self.delay_steps = delay_steps
        self.current_delay = self.delay_steps
        self.max_buffer_size = self.max_delay + 1
        self.observation_buffer = deque(maxlen=self.max_buffer_size)
        self.time_buffer = deque(maxlen=self.max_buffer_size)

        # System parameters with randomization
        self.params = SystemParameters()
        self.params.dt = dt  # Set the base dt in parameters

    def reset(self, random_init=True):
        # Reset state tracking
        self.elapsed_time = 0.0
        self.dt_history = []
        self.step_count = 0

        # Randomize system parameters for this episode
        self.params.randomize(self.randomization_factor, 0.0)  # Don't randomize dt during reset

        # Randomize the delay if using variable delay
        if self.min_delay != self.max_delay:
            self.current_delay = np.random.randint(self.min_delay, self.max_delay + 1)
        else:
            self.current_delay = self.delay_steps

        # Initialize state with random variations if requested
        if random_init:
            self.state = np.array([
                np.random.uniform(-0.1, 0.1),  # theta
                np.random.uniform(-0.1, 0.1),  # alpha
                np.random.uniform(-0.05, 0.05),  # theta_dot
                np.random.uniform(-0.05, 0.05)  # alpha_dot
            ])
        else:
            self.state = np.array([0.0, 0.1, 0.0, 0.0])

        # Initialize observation buffer
        observation = self._get_observation_from_state(self.state)
        self.observation_buffer.clear()
        self.time_buffer.clear()

        for _ in range(self.max_buffer_size):
            self.observation_buffer.append(observation)
            self.time_buffer.append(self.elapsed_time)

        # Return delayed observation
        return self._get_observation()

    def _get_observation_from_state(self, state):
        # Convert raw state to observation
        theta, alpha, theta_dot, alpha_dot = state
        alpha_norm = normalize_angle(alpha + np.pi)

        return np.array([
            np.sin(theta), np.cos(theta),
            np.sin(alpha_norm), np.cos(alpha_norm),
            theta_dot / 10.0,
            alpha_dot / 10.0
        ])

    def _get_observation(self):
        # Return observation with the current episode's delay
        if len(self.observation_buffer) <= self.current_delay:
            return self.observation_buffer[0]  # Fallback if buffer isn't filled yet
        return self.observation_buffer[-(self.current_delay + 1)]  # Get appropriately delayed observation

    def step(self, action):
        # Get a randomized dt for this step
        dt = self.params.randomize(0.0, self.dt_randomization_factor).dt_current
        self.dt_history.append(dt)

        # Update elapsed time
        self.elapsed_time += dt

        # Convert normalized action to voltage - properly handle array to scalar conversion
        if isinstance(action, np.ndarray):
            voltage = float(action.item()) * self.max_voltage
        else:
            voltage = float(action) * self.max_voltage

        # RK4 integration with current dt
        self.state = self._rk4_step(self.state, voltage, dt)

        # Enforce limits
        self.state = enforce_theta_limits(self.state, self.THETA_MIN, self.THETA_MAX)

        # Add current observation to buffer
        current_obs = self._get_observation_from_state(self.state)
        self.observation_buffer.append(current_obs)
        self.time_buffer.append(self.elapsed_time)

        # Get reward
        reward = self._compute_reward()

        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_observation(), reward, done, {'dt': dt}

    def _dynamics_step(self, state, t, vm):
        """Calculate system dynamics with randomized parameters"""
        theta_m, theta_L, theta_m_dot, theta_L_dot = state[0], state[1], state[2], state[3]
        p = self.params  # Get the current randomized parameters

        # Check theta limits - implement hard stops
        if (theta_m >= self.THETA_MAX and theta_m_dot > 0) or (theta_m <= self.THETA_MIN and theta_m_dot < 0):
            theta_m_dot = 0.0  # Stop the arm motion at the limits

        # Apply dead zone
        if -0.2 <= vm <= 0.2:
            vm = 0.0

        # Motor torque calculation with randomized params
        im = (vm - p.Km * theta_m_dot) / p.Rm
        Tm = p.Km * im

        # Equations of motion coefficients
        M11 = p.mL * p.LA ** 2 + p.quarter_mL_LL_squared - p.quarter_mL_LL_squared * np.cos(theta_L) ** 2 + p.JA
        M12 = -p.half_mL_LL_LA * np.cos(theta_L)
        C1 = 0.5 * p.mL * p.LL ** 2 * np.sin(theta_L) * np.cos(theta_L) * theta_m_dot * theta_L_dot
        C2 = p.half_mL_LL_LA * np.sin(theta_L) * theta_L_dot ** 2

        M21 = p.half_mL_LL_LA * np.cos(theta_L)
        M22 = p.JL + p.quarter_mL_LL_squared
        C3 = -p.quarter_mL_LL_squared * np.cos(theta_L) * np.sin(theta_L) * theta_m_dot ** 2
        G = p.half_mL_LL_g * np.sin(theta_L)

        # Calculate determinant for matrix inversion
        det_M = M11 * M22 - M12 * M21

        # Handle near-singular matrix
        if abs(det_M) < 1e-10:
            theta_m_ddot, theta_L_ddot = 0, 0
        else:
            # Right-hand side of equations
            RHS1 = Tm - C1 - C2 - p.DA * theta_m_dot
            RHS2 = -G - p.DL * theta_L_dot - C3

            # Solve for accelerations
            theta_m_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
            theta_L_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M

        return np.array([theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot])

    def _rk4_step(self, state, vm, dt):
        """4th-order Runge-Kutta integrator step with specified dt"""
        state = enforce_theta_limits(state, self.THETA_MIN, self.THETA_MAX)

        k1 = self._dynamics_step(state, 0, vm)
        state_k2 = enforce_theta_limits(state + 0.5 * dt * k1, self.THETA_MIN, self.THETA_MAX)
        k2 = self._dynamics_step(state_k2, 0, vm)

        state_k3 = enforce_theta_limits(state + 0.5 * dt * k2, self.THETA_MIN, self.THETA_MAX)
        k3 = self._dynamics_step(state_k3, 0, vm)

        state_k4 = enforce_theta_limits(state + dt * k3, self.THETA_MIN, self.THETA_MAX)
        k4 = self._dynamics_step(state_k4, 0, vm)

        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return enforce_theta_limits(new_state, self.THETA_MIN, self.THETA_MAX)

    def _compute_reward(self):
        """Calculate reward based on current state"""
        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_norm = normalize_angle(alpha + np.pi)  # Normalized for upright position
        p = self.params

        # Component 1: Reward for pendulum being close to upright
        upright_reward = np.cos(alpha_norm)  # 1 when upright, -1 when downward

        # Component 2: Penalty for high velocities
        velocity_penalty = -0.01 * (theta_dot ** 2 + alpha_dot ** 2)

        # Component 3: Penalty for arm position away from center
        pos_penalty = -0.01 * theta ** 2

        # Component 4: Extra reward for being very close to upright and stable
        bonus = 5.0 if abs(alpha_norm) < 0.2 and abs(alpha_dot) < 1.0 else 0.0

        # Component 5: Penalty for hitting limits
        limit_penalty = -10.0 if (abs(theta - self.THETA_MAX) < 0.02 or abs(theta - self.THETA_MIN) < 0.02) else 0.0

        # Component 6: Energy management reward
        E = p.Mp_g_Lp * (1 - np.cos(alpha)) + 0.5 * p.Jp * alpha_dot ** 2  # Current energy
        E_ref = p.Mp_g_Lp  # Energy at upright position (target energy)
        E_diff = abs(E - E_ref)  # Difference from optimal energy
        energy_reward = 2.0 * np.exp(-0.5 * (E_diff / (0.2 * E_ref)) ** 2)

        return upright_reward + velocity_penalty + pos_penalty + bonus + limit_penalty + energy_reward


# ============= Neural Network Models =============
class Actor(nn.Module):
    """Policy network that outputs action distributions"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log std for continuous action
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.network(state)

        # Get mean and constrain it to [-1, 1]
        action_mean = torch.tanh(self.mean(features))

        # Get log standard deviation and clamp it
        action_log_std = self.log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)

        return action_mean, action_log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from normal distribution
        normal = Normal(mean, std)
        x = normal.rsample()  # Reparameterization trick

        # Constrain to [-1, 1]
        action = torch.tanh(x)

        # Calculate log probability
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """Value network that estimates Q-values"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture (for Twin Delayed DDPG)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2


# ============= Replay Buffer =============
class ReplayBuffer:
    """Experience replay buffer for storing transitions"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = map(np.array, zip(*[self.buffer[i] for i in batch]))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============= SAC Agent =============
class SACAgent:
    """Soft Actor-Critic agent implementation"""

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)

        # Copy parameters to target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        if automatic_entropy_tuning:
            self.log_alpha = self.log_alpha.to(self.device)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                # Use mean action (no exploration)
                action, _ = self.actor(state)
            else:
                # Sample action with exploration
                action, _ = self.actor.sample(state)

            return action.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        # Sample batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # Get current alpha value
        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # Update critic
        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_action, next_log_prob = self.actor.sample(next_state_batch)

            # Get target Q values
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)

            # Compute target with entropy
            target_q = target_q - alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions, log_probs = self.actor.sample(state_batch)
        q1, q2 = self.critic(state_batch, actions)
        min_q = torch.min(q1, q2)

        # Actor loss (maximize Q - alpha * log_prob)
        actor_loss = (alpha * log_probs - min_q).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update automatic entropy tuning parameter
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }


# ============= Visualization Functions =============
def plot_episode(episode, states, actions, dt, reward, balanced_time=None, params=None,
                 prefix="episode", delay_steps=None, env=None, randomized=False):
    """Generic plotting function for episode visualization"""
    # Extract components
    thetas = states[:, 0]
    alphas = states[:, 1]
    theta_dots = states[:, 2]
    alpha_dots = states[:, 3]

    # Normalize alpha for visualization
    alpha_normalized = np.array([normalize_angle(a + np.pi) for a in alphas])

    # Generate time array
    current_dt = dt
    if env and hasattr(env, 'base_dt'):
        current_dt = env.base_dt

    # Use dt_history if available
    if env and hasattr(env, 'dt_history') and len(env.dt_history) > 0:
        t = np.cumsum(env.dt_history)
    else:
        t = np.arange(len(states)) * current_dt

    # Get delay information
    delay_info = ""
    if env is not None and hasattr(env, 'current_delay'):
        delay_info = f"Delay: {env.current_delay} steps ({env.current_delay * current_dt * 1000:.1f}ms), "
    elif delay_steps is not None:
        delay_info = f"Delay: {delay_steps} steps ({delay_steps * current_dt * 1000:.1f}ms), "

    # Plot results
    plt.figure(figsize=(12, 10))

    # Plot arm angle
    plt.subplot(4, 1, 1)
    plt.plot(t, thetas, 'b-')
    plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
    plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
    plt.ylabel('Arm angle (rad)')

    # Generate title with parameters if available
    if randomized and params:
        if hasattr(params, 'Rm'):  # It's a SystemParameters object
            param_text = f"{delay_info}Rm={params.Rm:.2f}, Km={params.Km:.4f}, mL={params.mL:.4f}, LL={params.LL:.4f}"
        else:  # It's a dict
            param_text = f"{delay_info}Rm={params['Rm']:.2f}, Km={params['Km']:.4f}, mL={params['mL']:.4f}, LL={params['LL']:.4f}"
        bal_text = f" | Balanced: {balanced_time:.2f}s" if balanced_time is not None else ""
        plt.title(f'{prefix.capitalize()} {episode + 1} | Reward: {reward:.2f}{bal_text}\n{param_text}')
    else:
        bal_text = f" | Balanced: {balanced_time:.2f}s" if balanced_time is not None else ""
        plt.title(f'{prefix.capitalize()} {episode + 1} | Reward: {reward:.2f}{bal_text}\n{delay_info}')

    plt.legend()
    plt.grid(True)

    # Plot pendulum angle
    plt.subplot(4, 1, 2)
    plt.plot(t, alpha_normalized, 'g-')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Line at upright position
    plt.axhline(y=0.17, color='r', linestyle=':', alpha=0.5)  # Upper balanced threshold
    plt.axhline(y=-0.17, color='r', linestyle=':', alpha=0.5)  # Lower balanced threshold
    plt.ylabel('Pendulum angle (rad)')
    plt.grid(True)

    # Plot pendulum angular velocity
    plt.subplot(4, 1, 3)
    plt.plot(t, alpha_dots, 'g-')
    plt.ylabel('Pendulum velocity (rad/s)')
    plt.grid(True)

    # Plot control actions
    plt.subplot(4, 1, 4)
    plt.plot(t, actions * MAX_VOLTAGE, 'r-')  # Scale to actual voltage
    plt.xlabel('Time (s)')
    plt.ylabel('Control voltage (V)')
    plt.ylim([-MAX_VOLTAGE * 1.1, MAX_VOLTAGE * 1.1])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{prefix}_{episode + 1}.png")
    plt.close()


# ============= Training and Evaluation Functions =============
def train_or_continue(env, agent=None, model_path=None, critic_path=None, episodes=1000,
                      batch_size=8192, replay_buffer_size=200000, max_steps=1000, lr=3e-4):
    """Unified training function that handles both new training and continuing from a checkpoint"""
    is_continuing = agent is not None or model_path is not None

    # Starting time
    start_time = time()

    # Initialize metrics
    episode_rewards = []
    avg_rewards = []

    # Initialize agent if not provided
    if agent is None:
        state_dim = 6  # Observation space size
        action_dim = 1  # Motor voltage (normalized)

        if model_path:
            # Load pretrained model
            print(f"Loading pretrained model from {model_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            agent = SACAgent(state_dim, action_dim)
            agent.actor.load_state_dict(torch.load(model_path, map_location=device))

            if critic_path:
                agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
                # Copy to target network
                for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(param.data)

            # Adjust learning rate for fine-tuning
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] = lr
            if agent.automatic_entropy_tuning:
                for param_group in agent.alpha_optimizer.param_groups:
                    param_group['lr'] = lr
        else:
            # Create new agent
            agent = SACAgent(state_dim, action_dim)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # Pre-fill buffer with some experiences if continuing training
    if is_continuing and len(replay_buffer) < batch_size:
        print("Pre-filling replay buffer with initial experiences...")
        prefill_steps = min(500, batch_size)
        while len(replay_buffer) < prefill_steps:
            state = env.reset()
            for step in range(100):  # Collect some steps
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

    # Training loop
    for episode in range(episodes):
        state = env.reset()  # Randomize parameters for this episode
        episode_reward = 0

        # If we're going to plot this episode, prepare to collect state data
        plot_this_episode = (episode + 1) % 10 == 0
        if plot_this_episode:
            episode_states = []
            episode_actions = []

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # If plotting, store the state and action
            if plot_this_episode:
                episode_states.append(env.state.copy())
                episode_actions.append(action)

            state = next_state
            episode_reward += reward

            # Update parameters if enough samples
            if len(replay_buffer) > batch_size:
                agent.update_parameters(replay_buffer, batch_size)

            if done:
                break

        # Log progress
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f}")

            # Plot episode for visual progress tracking
            if plot_this_episode:
                balanced_time = calculate_balanced_time(np.array(episode_states), env.base_dt)
                plot_episode(
                    episode,
                    np.array(episode_states),
                    np.array(episode_actions),
                    env.base_dt,
                    episode_reward,
                    balanced_time,
                    env.params,
                    prefix="training",
                    randomized=True,
                    delay_steps=env.delay_steps,
                    env=env
                )

        # Early stopping if well trained
        if avg_reward > 5000 and episode > 200:  # Higher threshold for robust policy
            print(f"Environment solved in {episode + 1} episodes!")
            break

    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Save trained model
    timestamp = int(time())
    prefix = "continued" if is_continuing else "sac"
    torch.save(agent.actor.state_dict(), f"{prefix}_actor_{timestamp}.pth")
    torch.save(agent.critic.state_dict(), f"{prefix}_critic_{timestamp}.pth")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'{"Continued" if is_continuing else "SAC"} Training Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_training_{timestamp}.png")
    plt.close()

    return agent


def load_model(model_path, critic_path=None, state_dim=6, action_dim=1, hidden_dim=256):
    """Load a pre-trained SAC model"""
    print(f"Loading model from {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a new agent
    agent = SACAgent(state_dim, action_dim, hidden_dim)

    # Load actor state dict
    agent.actor.load_state_dict(torch.load(model_path, map_location=device))
    agent.actor.eval()  # Set to evaluation mode

    # Load critic if provided
    if critic_path:
        agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
        agent.critic.eval()

        # Copy critic weights to target critic
        for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
            target_param.data.copy_(param.data)

    print("Model loaded successfully!")
    return agent


def calculate_balanced_time(states, dt):
    """Calculate time spent with pendulum balanced"""
    balanced_time = 0.0
    for state in states:
        alpha_norm = normalize_angle(state[1] + np.pi)
        if abs(alpha_norm) < 0.17:  # ~10 degrees from vertical
            balanced_time += dt
    return balanced_time


def evaluate_agent(agent, env, num_episodes=3, prefix="evaluation"):
    """Evaluate an agent's performance in an environment"""
    performance = {
        'rewards': [],
        'balanced_times': [],
        'params_used': [] if hasattr(env, 'params') else None
    }

    for episode in range(num_episodes):
        # Reset environment
        state = env.reset(random_init=True)
        total_reward = 0

        # Store episode data
        states_history = []
        actions_history = []

        # Remember parameters for this episode if using randomization
        if hasattr(env, 'params'):
            performance['params_used'].append({
                'Rm': env.params.Rm,
                'Km': env.params.Km,
                'mA': env.params.mA,
                'mL': env.params.mL,
                'LA': env.params.LA,
                'LL': env.params.LL,
                'g': env.params.g
            })

        for step in range(env.max_steps):
            # Get deterministic action
            action = agent.select_action(state, evaluate=True)

            # Take step in environment
            next_state, reward, done, _ = env.step(action)

            # Record data
            states_history.append(env.state.copy())
            actions_history.append(action)

            # Update metrics
            total_reward += reward
            state = next_state

            if done:
                break

        # Calculate balancing performance
        balanced_time = calculate_balanced_time(np.array(states_history), env.base_dt)

        # Store performance metrics
        performance['rewards'].append(total_reward)
        performance['balanced_times'].append(balanced_time)

        # Print results
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Balanced = {balanced_time:.2f}s")

        # Create visualization
        plot_episode(
            episode,
            np.array(states_history),
            np.array(actions_history),
            env.base_dt,
            total_reward,
            balanced_time,
            env.params if hasattr(env, 'params') else None,
            prefix=prefix,
            randomized=hasattr(env, 'randomization_factor') and env.randomization_factor > 0,
            delay_steps=env.delay_steps if hasattr(env, 'delay_steps') else None,
            env=env
        )

    # Create summary plot if multiple episodes
    if num_episodes > 1:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, num_episodes + 1), performance['rewards'])
        plt.axhline(y=np.mean(performance['rewards']), color='r', linestyle='--',
                    label=f'Mean: {np.mean(performance["rewards"]):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.bar(range(1, num_episodes + 1), performance['balanced_times'])
        plt.axhline(y=np.mean(performance['balanced_times']), color='r', linestyle='--',
                    label=f'Mean: {np.mean(performance["balanced_times"]):.2f}s')
        plt.xlabel('Episode')
        plt.ylabel('Time Balanced (s)')
        plt.title('Balancing Performance')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{prefix}_summary.png")
        plt.close()

    return performance


def evaluate_robustness(agent, base_env, randomization_levels=[0.0, 0.05, 0.1, 0.2],
                        num_episodes=3):
    """Evaluate agent's robustness across different randomization levels"""
    results = {}

    for rand_level in randomization_levels:
        print(f"\nEvaluating with {rand_level * 100}% parameter randomization...")

        # Create new environment with current randomization level
        env = PendulumEnv(
            dt=base_env.base_dt,
            randomization_factor=rand_level,
            delay_steps=base_env.delay_steps if hasattr(base_env, 'delay_steps') else 0,
            delay_range=(base_env.min_delay, base_env.max_delay) if hasattr(base_env, 'min_delay') else None,
            dt_randomization_factor=base_env.dt_randomization_factor if hasattr(base_env,
                                                                                'dt_randomization_factor') else 0
        )

        # Evaluate performance
        performance = evaluate_agent(
            agent,
            env,
            num_episodes=num_episodes,
            prefix=f"robustness_{int(rand_level * 100)}"
        )

        # Store results
        results[rand_level] = {
            'mean_reward': np.mean(performance['rewards']),
            'std_reward': np.std(performance['rewards']),
            'mean_balanced_time': np.mean(performance['balanced_times']),
            'std_balanced_time': np.std(performance['balanced_times'])
        }

    # Plot summary of robustness results
    plt.figure(figsize=(12, 6))

    # Convert randomization levels to percentages for plotting
    x_labels = [f"{r * 100}%" for r in randomization_levels]

    # Plot rewards
    plt.subplot(1, 2, 1)
    means = [results[r]['mean_reward'] for r in randomization_levels]
    stds = [results[r]['std_reward'] for r in randomization_levels]
    plt.bar(x_labels, means, yerr=stds, capsize=10)
    plt.title('Reward vs. Parameter Randomization')
    plt.xlabel('Randomization Level')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)

    # Plot balanced times
    plt.subplot(1, 2, 2)
    means = [results[r]['mean_balanced_time'] for r in randomization_levels]
    stds = [results[r]['std_balanced_time'] for r in randomization_levels]
    plt.bar(x_labels, means, yerr=stds, capsize=10)
    plt.title('Balancing Time vs. Parameter Randomization')
    plt.xlabel('Randomization Level')
    plt.ylabel('Average Time Balanced (s)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("robustness_summary.png")
    plt.close()

    return results


# ============= Main Function =============
def main():
    parser = argparse.ArgumentParser(description='Pendulum RL Training and Evaluation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train from scratch
    train_parser = subparsers.add_parser('train', help='Train a new model from scratch')
    train_parser.add_argument('--random', type=float, default=0.0, help='Randomization factor (0.0-1.0)')
    train_parser.add_argument('--delay', type=int, default=0, help='Fixed observation delay steps')
    train_parser.add_argument('--delay-range', type=str, help='Variable delay range, format: min-max (e.g., 0-5)')
    train_parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for continued training')
    train_parser.add_argument('--eval', action='store_true', help='Evaluate the model after training')
    train_parser.add_argument('--dt', type=float, default=0.0115, help='Base timestep in seconds')
    train_parser.add_argument('--dt-random', type=float, default=0.0, help='Randomization factor for timestep')

    # Load and evaluate a model
    load_parser = subparsers.add_parser('load', help='Load and evaluate a pre-trained model')
    load_parser.add_argument('model_path', help='Path to the saved model file')
    load_parser.add_argument('--random', type=float, default=0.0, help='Randomization factor for evaluation')
    load_parser.add_argument('--delay', type=int, default=0, help='Fixed observation delay steps')
    load_parser.add_argument('--delay-range', type=str, help='Variable delay range, format: min-max (e.g., 0-5)')
    load_parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to evaluate')
    load_parser.add_argument('--dt', type=float, default=0.0115, help='Base timestep in seconds')
    load_parser.add_argument('--dt-random', type=float, default=0.0, help='Randomization factor for timestep')

    # Continue training a model
    continue_parser = subparsers.add_parser('continue', help='Continue training a pre-trained model')
    continue_parser.add_argument('model_path', help='Path to the saved actor model file')
    continue_parser.add_argument('--critic', help='Optional path to the saved critic model file')
    continue_parser.add_argument('--random', type=float, default=0.0, help='Randomization factor for training')
    continue_parser.add_argument('--delay', type=int, default=0, help='Fixed observation delay steps')
    continue_parser.add_argument('--delay-range', type=str, help='Variable delay range, format: min-max (e.g., 0-5)')
    continue_parser.add_argument('--episodes', type=int, default=500, help='Number of additional episodes to train')
    continue_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for continued training')
    continue_parser.add_argument('--eval', action='store_true', help='Evaluate the model after training')
    continue_parser.add_argument('--dt', type=float, default=0.0115, help='Base timestep in seconds')
    continue_parser.add_argument('--dt-random', type=float, default=0.0, help='Randomization factor for timestep')

    args = parser.parse_args()

    # Default to train if no command specified
    if not args.command:
        args.command = 'train'

    # Parse delay range if provided
    delay_range = None
    if hasattr(args, 'delay_range') and args.delay_range:
        try:
            min_delay, max_delay = map(int, args.delay_range.split('-'))
            delay_range = (min_delay, max_delay)
            print(f"Using variable delay in range {delay_range}")
        except ValueError:
            print(f"Warning: Invalid delay range format '{args.delay_range}'. Using fixed delay of {args.delay}.")

    # Create environment with appropriate parameters
    env = PendulumEnv(
        dt=args.dt,
        randomization_factor=args.random,
        delay_steps=args.delay,
        delay_range=delay_range,
        dt_randomization_factor=args.dt_random if hasattr(args, 'dt_random') else 0.0
    )

    # Execute the appropriate command
    if args.command == 'train':
        print(f"\nTraining new model from scratch...")
        print(f"Parameters: randomization={args.random}, delay={args.delay if not delay_range else delay_range}")

        agent = train_or_continue(
            env,
            episodes=args.episodes,
            lr=args.lr
        )

        # Evaluate trained agent
        print("\nEvaluating trained agent...")
        evaluate_robustness(
            agent,
            env,
            num_episodes=2,
            randomization_levels=[0.0, args.random, args.random * 2]
        )

        # Optionally evaluate the improved agent
        if args.eval:
            print("\nEvaluating improved agent...")
            evaluate_robustness(
                agent,
                env,
                num_episodes=2,
                randomization_levels=[0.0, args.random, args.random * 2]
            )

    elif args.command == 'load':
        print(f"\nRunning pre-trained model from {args.model_path}...")

        agent = load_model(args.model_path)
        performance = evaluate_agent(
            agent,
            env,
            num_episodes=args.episodes,
            prefix="evaluation"
        )

        # Print performance statistics
        avg_reward = np.mean(performance['rewards'])
        avg_balanced = np.mean(performance['balanced_times'])
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average balancing time: {avg_balanced:.2f}s")

    elif args.command == 'continue':
        print(f"\nContinuing training from model: {args.model_path}")

        agent = train_or_continue(
            env,
            model_path=args.model_path,
            critic_path=args.critic,
            episodes=args.episodes,
            lr=args.lr
        )

        # Optionally evaluate the improved agent
        if args.eval:
            print("\nEvaluating improved agent...")
            evaluate_robustness(
                agent,
                env,
                num_episodes=2,
                randomization_levels=[0.0, args.random, args.random * 2]
            )

    print("=" * 50)
    print("PROGRAM EXECUTION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()