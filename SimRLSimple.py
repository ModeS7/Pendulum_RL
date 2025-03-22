import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb
from time import time

# Import the necessary constants and functions from your existing code
# System Parameters (same as in your original code)
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

max_voltage = 10.0  # Maximum motor voltage
THETA_MIN = -2.2  # Minimum arm angle (radians)
THETA_MAX = 2.2  # Maximum arm angle (radians)

# Pre-compute constants for optimization (same as in your original code)
half_mL_LL_g = 0.5 * mL * LL * g
half_mL_LL_LA = 0.5 * mL * LL * LA
quarter_mL_LL_squared = 0.25 * mL * LL ** 2
Mp_g_Lp = mL * g * LL
Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia (kg·m²)

batch_size = 256 * 32  # Batch size for training


# Reuse some of your helper functions
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
    """Ultra-optimized dynamics calculation with theta limits"""
    theta_m, theta_L, theta_m_dot, theta_L_dot = state[0], state[1], state[2], state[3]

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


# Simulation environment for RL
class PendulumEnv:
    def __init__(self, dt=0.0115, max_steps=800):  # 15 seconds at 50Hz
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.state = None

    def reset(self, random_init=True):
        # Initialize with small random variations if requested
        if random_init:
            self.state = np.array([
                np.random.uniform(-0.1, 0.1),  # theta
                np.random.uniform(-0.1, 0.1),  # alpha (small angle from bottom)
                np.random.uniform(-0.05, 0.05),  # theta_dot
                np.random.uniform(-0.05, 0.05)  # alpha_dot
            ])
        else:
            self.state = np.array([0.0, 0.1, 0.0, 0.0])  # Default initial state

        self.step_count = 0
        return self._get_observation()

    def _get_observation(self):
        # Normalize the state for RL
        theta, alpha, theta_dot, alpha_dot = self.state

        # Normalize angles to [-π, π] for the observation
        alpha_norm = normalize_angle(alpha + np.pi)  # Normalized relative to upright

        # Create observation vector with sin/cos values for angles to avoid discontinuities
        obs = np.array([
            np.sin(theta), np.cos(theta),
            np.sin(alpha_norm), np.cos(alpha_norm),
            theta_dot / 10.0,  # Scale velocities to roughly [-1, 1]
            alpha_dot / 10.0
        ])

        return obs

    def step(self, action):
        # Convert normalized action [-1, 1] to voltage
        voltage = float(action) * max_voltage

        # RK4 integration step
        self.state = self._rk4_step(self.state, voltage)

        # Enforce limits
        self.state = enforce_theta_limits(self.state)

        # Get reward
        reward = self._compute_reward()

        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_observation(), reward, done, {}

    def _rk4_step(self, state, vm):
        """4th-order Runge-Kutta integrator step"""
        # Apply limits to initial state
        state = enforce_theta_limits(state)

        k1 = dynamics_step(state, 0, vm)
        state_k2 = enforce_theta_limits(state + 0.5 * self.dt * k1)
        k2 = dynamics_step(state_k2, 0, vm)

        state_k3 = enforce_theta_limits(state + 0.5 * self.dt * k2)
        k3 = dynamics_step(state_k3, 0, vm)

        state_k4 = enforce_theta_limits(state + self.dt * k3)
        k4 = dynamics_step(state_k4, 0, vm)

        new_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return enforce_theta_limits(new_state)

    def _compute_reward(self):
        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_norm = normalize_angle(alpha + np.pi)  # Normalized for upright position

        # Component 1: Reward for pendulum being close to upright
        upright_reward = np.cos(alpha_norm)  # 1 when upright, -1 when downward

        # Component 2: Penalty for high velocities
        velocity_penalty = -0.01 * (theta_dot ** 2 + alpha_dot ** 2)

        # Component 3: Penalty for arm position away from center
        pos_penalty = -0.01 * theta ** 2

        # Component 4: Extra reward for being very close to upright and stable
        bonus = 0.0
        if abs(alpha_norm) < 0.2 and abs(alpha_dot) < 1.0:
            bonus = 5.0

        # Component 5: Penalty for hitting limits
        limit_penalty = 0.0
        if abs(theta - THETA_MAX) < 0.01 or abs(theta - THETA_MIN) < 0.01:
            limit_penalty = -10.0

        # Component 6: Energy management reward
        E = Mp_g_Lp * (1 - np.cos(alpha)) + 0.5 * Jp * alpha_dot ** 2  # Current energy
        E_ref = Mp_g_Lp  # Energy at upright position (target energy)
        E_diff = abs(E - E_ref)  # Difference from optimal energy
        # Reward for being close to the optimal energy (inverted Gaussian)
        energy_reward = 2.0 * np.exp(-0.5 * (E_diff / (0.2 * E_ref)) ** 2)

        return upright_reward + bonus + limit_penalty + energy_reward


# Actor (Policy) Network
class Actor(nn.Module):
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


# Critic (Value) Network
class Critic(nn.Module):
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


# Soft Actor-Critic (SAC) Agent
class SACAgent:
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

        if evaluate:
            # Use mean action (no exploration)
            with torch.no_grad():
                action, _ = self.actor(state)
                return action.cpu().numpy()[0]
        else:
            # Sample action with exploration
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size=batch_size):
        # Sample batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_action, next_log_prob = self.actor.sample(next_state_batch)

            # Get target Q values
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)

            # Compute target with entropy
            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

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

        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

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


# Replay Buffer
class ReplayBuffer:
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


def train():
    print("Starting SAC training for inverted pendulum control...")

    # Environment setup
    env = PendulumEnv()

    # Hyperparameters
    state_dim = 6  # Our observation space
    action_dim = 1  # Motor voltage (normalized)
    max_episodes = 100
    max_steps = 800
    batch_size = 256
    replay_buffer_size = 100000
    updates_per_step = 1

    # Initialize agent
    agent = SACAgent(state_dim, action_dim)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # Metrics
    episode_rewards = []
    avg_rewards = []

    # For visualization - store episode data
    states_history = []
    actions_history = []

    # Training loop
    start_time = time()

    for episode in range(max_episodes):
        state = env.reset()
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

            # If we're plotting, store the raw state and action
            if plot_this_episode:
                episode_states.append(env.state.copy())
                episode_actions.append(action)

            state = next_state
            episode_reward += reward

            # Update if enough samples
            if len(replay_buffer) > batch_size:
                for _ in range(updates_per_step):
                    agent.update_parameters(replay_buffer, batch_size)

            if done:
                break

        # Log progress
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{max_episodes} | Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f}")

            # Plot simulation for visual progress tracking
            if plot_this_episode:
                plot_training_episode(episode, episode_states, episode_actions, env.dt, episode_reward)

        # Early stopping if well trained
        if avg_reward > 5000 and episode > 100:
            print(f"Environment solved in {episode + 1} episodes!")
            break

    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Save trained model
    timestamp = int(time())
    torch.save(agent.actor.state_dict(), f"actor_{timestamp}.pth")
    torch.save(agent.critic.state_dict(), f"critic_{timestamp}.pth")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='100-Episode Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('SAC Training for Inverted Pendulum')
    plt.legend()
    plt.grid(True)
    plt.savefig("sac_training_progress.png")
    plt.show()

    return agent


def plot_training_episode(episode, states_history, actions_history, dt, episode_reward):
    """Plot the pendulum state evolution for a training episode"""
    # Convert history to numpy arrays
    states_history = np.array(states_history)
    actions_history = np.array(actions_history)

    # Extract components
    thetas = states_history[:, 0]
    alphas = states_history[:, 1]
    theta_dots = states_history[:, 2]
    alpha_dots = states_history[:, 3]

    # Normalize alpha for visualization
    alpha_normalized = np.zeros(len(alphas))
    for i in range(len(alphas)):
        alpha_normalized[i] = normalize_angle(alphas[i] + np.pi)

    # Generate time array
    t = np.arange(len(states_history)) * dt

    # Count performance metrics
    balanced_time = 0.0
    num_upright_points = 0

    for i in range(len(alpha_normalized)):
        # Check if pendulum is close to upright
        if abs(alpha_normalized[i]) < 0.17:  # about 10 degrees
            balanced_time += dt
            num_upright_points += 1

    # Plot results
    plt.figure(figsize=(12, 9))

    # Plot arm angle
    plt.subplot(3, 1, 1)
    plt.plot(t, thetas, 'b-')
    plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
    plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
    plt.ylabel('Arm angle (rad)')
    plt.title(f'Training Episode {episode} | Reward: {episode_reward:.2f} | Balanced: {balanced_time:.2f}s')
    plt.legend()
    plt.grid(True)

    # Plot pendulum angle
    plt.subplot(3, 1, 2)
    plt.plot(t, alpha_normalized, 'g-')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Line at upright position
    plt.axhline(y=0.17, color='r', linestyle=':', alpha=0.5)  # Upper balanced threshold
    plt.axhline(y=-0.17, color='r', linestyle=':', alpha=0.5)  # Lower balanced threshold
    plt.ylabel('Pendulum angle (rad)')
    plt.grid(True)

    # Plot control actions
    plt.subplot(3, 1, 3)
    plt.plot(t, actions_history * max_voltage, 'r-')  # Scale back to actual voltage
    plt.xlabel('Time (s)')
    plt.ylabel('Control voltage (V)')
    plt.ylim([-max_voltage * 1.1, max_voltage * 1.1])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"training_episode_{episode}.png")
    #plt.close()  # Close the figure to avoid memory issues


def evaluate(agent, num_episodes=5, render=True):
    """Evaluate the trained agent's performance"""
    env = PendulumEnv()

    for episode in range(num_episodes):
        state = env.reset(random_init=False)  # Start from standard position
        total_reward = 0

        # Store episode data for plotting
        states_history = []
        actions_history = []

        for step in range(env.max_steps):
            # Select action
            action = agent.select_action(state, evaluate=True)

            # Step environment
            next_state, reward, done, _ = env.step(action)

            # Store data
            states_history.append(env.state.copy())  # Store raw state
            actions_history.append(action)

            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

        if render:
            # Convert history to numpy arrays
            states_history = np.array(states_history)
            actions_history = np.array(actions_history)

            # Extract components
            thetas = states_history[:, 0]
            alphas = states_history[:, 1]
            theta_dots = states_history[:, 2]
            alpha_dots = states_history[:, 3]

            # Normalize alpha for visualization
            alpha_normalized = np.zeros(len(alphas))
            for i in range(len(alphas)):
                alpha_normalized[i] = normalize_angle(alphas[i] + np.pi)

            # Generate time array
            t = np.arange(len(states_history)) * env.dt

            # Count performance metrics
            balanced_time = 0.0
            num_upright_points = 0

            for i in range(len(alpha_normalized)):
                # Check if pendulum is close to upright
                if abs(alpha_normalized[i]) < 0.17:  # about 10 degrees
                    balanced_time += env.dt
                    num_upright_points += 1

            # Plot results
            plt.figure(figsize=(14, 12))

            # Plot arm angle
            plt.subplot(4, 1, 1)
            plt.plot(t, thetas, 'b-')
            plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
            plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
            plt.ylabel('Arm angle (rad)')
            plt.title(f'SAC Inverted Pendulum Control - Episode {episode + 1}')
            plt.legend()
            plt.grid(True)

            # Plot pendulum angle
            plt.subplot(4, 1, 2)
            plt.plot(t, alpha_normalized, 'g-')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Line at upright position
            plt.ylabel('Pendulum angle (rad)')
            plt.grid(True)

            # Plot motor voltage (actions)
            plt.subplot(4, 1, 3)
            plt.plot(t, actions_history * max_voltage, 'r-')  # Scale back to actual voltage
            plt.ylabel('Control voltage (V)')
            plt.ylim([-max_voltage * 1.1, max_voltage * 1.1])
            plt.grid(True)

            # Plot angular velocities
            plt.subplot(4, 1, 4)
            plt.plot(t, theta_dots, 'b-', label='Arm velocity')
            plt.plot(t, alpha_dots, 'g-', label='Pendulum velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Angular velocity (rad/s)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"sac_evaluation_episode_{episode + 1}.png")
            plt.show()

            print(f"Time spent balanced: {balanced_time:.2f} seconds")
            print(f"Data points with pendulum upright: {num_upright_points}")
            print(f"Max arm angle: {np.max(np.abs(thetas)):.2f} rad")
            print(f"Max pendulum angular velocity: {np.max(np.abs(alpha_dots)):.2f} rad/s")
            final_angle_deg = abs(alpha_normalized[-1]) * 180 / np.pi
            print(
                f"Final pendulum angle from vertical: {abs(alpha_normalized[-1]):.2f} rad ({final_angle_deg:.1f} degrees)")
            print("-" * 50)


if __name__ == "__main__":
    print("TorchRL Inverted Pendulum Training and Evaluation")
    print("=" * 50)

    # Train agent
    agent = train()

    # Evaluate trained agent
    print("\nEvaluating trained agent...")
    evaluate(agent, num_episodes=3)

    print("=" * 50)
    print("PROGRAM EXECUTION COMPLETE")
    print("=" * 50)