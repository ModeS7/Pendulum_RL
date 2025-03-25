import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
# Remove numba import as we're not using it now
# import numba as nb
from collections import deque
import matplotlib.pyplot as plt
import time

# System Parameters
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

# Pre-compute constants for optimization
half_mL_LL_g = 0.5 * mL * LL * g
half_mL_LL_LA = 0.5 * mL * LL * LA
quarter_mL_LL_squared = 0.25 * mL * LL ** 2
Mp_g_Lp = mL * g * LL
Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia (kg·m²)

batch_size = 256 * 32  # Batch size for training


# Helper functions
# No njit for this function since we need to handle both arrays and scalars
def clip_value(value, min_value, max_value):
    """Clip function that works with both arrays and scalars"""
    return np.clip(value, min_value, max_value)


# Remove njit for better type compatibility
def apply_voltage_deadzone(vm):
    """Apply motor voltage dead zone"""
    # Handle both scalar and array inputs
    return np.where(np.abs(vm) <= 0.2, 0.0, vm)


# Remove njit for type compatibility
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


# Remove njit for type compatibility
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

    return np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)


# Remove njit to handle mixed types better
def dynamics_step(state, t, vm):
    """Dynamics calculation with theta limits"""
    # Convert input to float scalar values to avoid array issues
    theta_m = float(state[0])
    theta_L = float(state[1])
    theta_m_dot = float(state[2])
    theta_L_dot = float(state[3])

    # If vm is an array with one element, convert to scalar
    if hasattr(vm, 'shape') and vm.size == 1:
        vm = float(vm)

    # Check theta limits - implement hard stops
    if (theta_m >= THETA_MAX and theta_m_dot > 0) or (theta_m <= THETA_MIN and theta_m_dot < 0):
        theta_m_dot = 0.0  # Stop the arm motion at the limits

    # Apply dead zone and calculate motor torque
    if -0.2 <= vm <= 0.2:  # Direct implementation rather than using the function
        vm = 0.0

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
        theta_m_ddot = 0.0
        theta_L_ddot = 0.0
    else:
        # Right-hand side of equations
        RHS1 = Tm - C1 - C2 - DA * theta_m_dot
        RHS2 = -G - DL * theta_L_dot - C3

        # Solve for accelerations
        theta_m_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
        theta_L_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M

    # Create a numpy array with all values as float32
    return np.array([theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot], dtype=np.float32)


# Neural Network Architecture for PPO
class Actor(nn.Module):
    """Policy network for the PPO algorithm"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Mean and log_std heads for the action distribution
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Forward pass through the network"""
        x = self.network(state)
        mean = self.mean(x)

        # Get the action distribution
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)

        return dist


class Critic(nn.Module):
    """Value network for the PPO algorithm"""

    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """Forward pass through the network"""
        return self.network(state)


class PPOBuffer:
    """Storage buffer for PPO algorithm"""

    def __init__(self, state_dim, action_dim, buffer_size, gamma=0.99, lam=0.95):
        self.state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.next_state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.value_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logp_buf = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.return_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buffer_size

    def store(self, state, action, reward, next_state, value, logp, done):
        """Store a transition in the buffer"""
        assert self.ptr < self.max_size

        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_state_buf[self.ptr] = next_state
        self.value_buf[self.ptr] = value
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done

        self.ptr += 1

    def finish_path(self, last_val=0):
        """Compute advantage estimates when an episode is finished"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buf[path_slice], last_val)
        values = np.append(self.value_buf[path_slice], last_val)

        # GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - self.done_buf[path_slice]) - values[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # Compute returns for TD(λ)
        self.return_buf[path_slice] = self._discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sum (for returns and advantages)"""
        n = len(x)
        y = np.zeros_like(x)
        y[n - 1] = x[n - 1]
        for t in reversed(range(n - 1)):
            y[t] = x[t] + discount * y[t + 1]
        return y

    def get(self):
        """Get all stored data from the buffer"""
        assert self.ptr == self.max_size  # Buffer must be full before using
        self.ptr, self.path_start_idx = 0, 0

        # Normalize advantages
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        return dict(
            states=self.state_buf,
            actions=self.action_buf,
            returns=self.return_buf,
            advantages=self.adv_buf,
            logp_old=self.logp_buf
        )


class PPOAgent:
    """PPO agent implementation"""

    def __init__(self, state_dim, action_dim, action_low, action_high,
                 gamma=0.99, clip_ratio=0.2, policy_lr=3e-4, vf_lr=1e-3,
                 train_iters=80, target_kl=0.01, hidden_dim=64, lam=0.97):

        # Environment parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        # PPO hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.lam = lam

        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=vf_lr)

        # For tracking training progress
        self.episode_rewards = []
        self.mean_rewards = []

    def get_action(self, state, deterministic=False):
        """Get action from policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        dist = self.actor(state_tensor)

        if deterministic:
            action = dist.mean.detach().numpy()[0]
        else:
            action = dist.sample().detach().numpy()[0]

        # Clip action to environment's action space
        action = np.clip(action, self.action_low, self.action_high)

        # Get log probability of the action
        log_prob = dist.log_prob(torch.FloatTensor(action)).sum().detach().numpy()

        # Get value estimate
        value = self.critic(state_tensor).detach().numpy()[0, 0]

        return action, log_prob, value

    def update(self, buffer):
        """Update the policy and value network using PPO"""
        data = buffer.get()

        states = torch.FloatTensor(data['states'])
        actions = torch.FloatTensor(data['actions'])
        returns = torch.FloatTensor(data['returns']).unsqueeze(1)
        advantages = torch.FloatTensor(data['advantages']).unsqueeze(1)
        old_logp = torch.FloatTensor(data['logp_old']).unsqueeze(1)

        # Train the policy and value network multiple times
        for _ in range(self.train_iters):
            # Get current action distribution and values
            dist = self.actor(states)
            values = self.critic(states)

            # Calculate log probabilities of actions
            logp = dist.log_prob(actions).sum(dim=1, keepdim=True)

            # Calculate ratio for PPO
            ratio = torch.exp(logp - old_logp)

            # Calculate surrogate losses
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # Policy loss (negative because we're maximizing)
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value function loss
            value_loss = ((values - returns) ** 2).mean()

            # Calculate approximate KL divergence for early stopping
            approx_kl = ((old_logp - logp) ** 2).mean()

            # Update the policy network
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # Update the value network
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            # Stop early if KL divergence is too large
            if approx_kl > 1.5 * self.target_kl:
                break

    def save(self, path):
        """Save the model weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path):
        """Load the model weights"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def train(self, env, num_epochs=100, steps_per_epoch=4000, batch_size=64, max_ep_len=1000, render_every=10):
        """Train the agent using PPO"""
        # Initialize replay buffer
        buffer = PPOBuffer(self.state_dim, self.action_dim, steps_per_epoch, self.gamma, self.lam)

        # Track average reward
        best_mean_reward = -np.inf

        # Start training
        total_steps = 0
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Collect trajectories
            state = env.reset()
            episode_reward = 0
            episode_length = 0

            for t in range(steps_per_epoch):
                # Get action
                action, logp, val = self.get_action(state)

                # Take step in environment
                next_state, reward, done, _ = env.step(action)

                # Store transition in buffer
                buffer.store(state, action, reward, next_state, val, logp, done)

                # Update state and counters
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1

                # Handle episode termination
                timeout = episode_length >= max_ep_len
                terminal = done or timeout

                if terminal or t == steps_per_epoch - 1:
                    if not terminal:
                        # If trajectory didn't reach terminal state, bootstrap value
                        _, _, last_val = self.get_action(state)
                    else:
                        last_val = 0

                    buffer.finish_path(last_val)

                    # Only record episode reward if it's a genuine episode termination
                    if terminal:
                        self.episode_rewards.append(episode_reward)

                    # Reset for next episode
                    state = env.reset()
                    episode_reward = 0
                    episode_length = 0

            # Update the policy and value networks after collecting a batch of data
            self.update(buffer)

            # Calculate mean reward over last 10 episodes
            mean_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            self.mean_rewards.append(mean_reward)

            # Save model if it's the best so far
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                self.save('best_ppo_model.pt')

            # Render every few epochs
            if epoch % render_every == 0:
                test_env = PendulumEnv()
                test_state = test_env.reset()
                done = False

                while not done:
                    action, _, _ = self.get_action(test_state, deterministic=True)
                    test_state, _, done, _ = test_env.step(action)

                test_env.render()

            # Print epoch statistics
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{num_epochs} | Mean Reward: {mean_reward:.2f} | Time: {epoch_time:.2f}s")

        # Plot learning curve
        plt.figure(figsize=(10, 5))
        plt.plot(self.mean_rewards)
        plt.title("Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Reward")
        plt.savefig("learning_curve.png")
        plt.show()

        return self.mean_rewards


# Simulation environment for RL
class PendulumEnv:
    def __init__(self, dt=0.01, max_steps=500):
        self.dt = dt  # Time step for simulation
        self.max_steps = max_steps
        self.current_steps = 0
        self.state = None

        # Action space: motor voltage
        self.action_space_low = -max_voltage
        self.action_space_high = max_voltage

        # State space dimensions
        # For normalized observation, dimension is 6 (sin/cos of angles + velocities)
        # For raw state, dimension is 4 (theta, alpha, theta_dot, alpha_dot)
        self.observation_dim = 6  # Using normalized observation with sin/cos

        # For rendering
        self.state_history = []
        self.reward_history = []
        self.action_history = []

    def reset(self, random_init=True):
        """Reset the environment"""
        # Initialize state
        if random_init:
            # Small random variations for more robust training
            self.state = np.array([
                np.random.uniform(-0.1, 0.1),  # theta
                np.pi + np.random.uniform(-0.1, 0.1),  # alpha (near bottom)
                np.random.uniform(-0.05, 0.05),  # theta_dot
                np.random.uniform(-0.05, 0.05)  # alpha_dot
            ], dtype=np.float32)
        else:
            # Default initial state (pendulum hanging down)
            self.state = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)

        self.current_steps = 0
        self.state_history = [self.state.copy()]
        self.reward_history = []
        self.action_history = []

        return self._get_observation()

    def _get_observation(self):
        """Convert internal state to observation for RL agent"""
        theta, alpha, theta_dot, alpha_dot = self.state

        # Normalize angles using sin/cos to avoid discontinuities
        # This is important for the policy to learn smoother representations
        obs = np.array([
            np.sin(theta), np.cos(theta),  # Arm angle
            np.sin(alpha), np.cos(alpha),  # Pendulum angle
            theta_dot / 10.0,  # Normalize velocities
            alpha_dot / 10.0  # to be roughly in [-1, 1]
        ], dtype=np.float32)

        return obs

    def step(self, action):
        """Take a step in the environment with the given action"""
        # Clip action to valid range
        vm = clip_value(action, -max_voltage, max_voltage)

        # Store action for visualization
        self.action_history.append(vm)

        # Integrate using RK4 for better accuracy
        state_array = self.state.astype(np.float32)  # Ensure consistent type
        k1 = dynamics_step(state_array, 0, vm)
        k2 = dynamics_step(state_array + 0.5 * self.dt * k1, 0, vm)
        k3 = dynamics_step(state_array + 0.5 * self.dt * k2, 0, vm)
        k4 = dynamics_step(state_array + self.dt * k3, 0, vm)

        # Update state
        new_state = self.state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce limits on theta
        new_state = enforce_theta_limits(new_state)

        # Normalize alpha to [-π, π]
        new_state[1] = normalize_angle(new_state[1])

        self.state = new_state
        self.current_steps += 1

        # Store state for visualization
        self.state_history.append(self.state.copy())

        # Calculate reward
        reward = self._compute_reward()
        self.reward_history.append(reward)

        # Check termination
        done = self.current_steps >= self.max_steps

        return self._get_observation(), reward, done, {}

    def _compute_reward(self):
        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_norm = normalize_angle(alpha + np.pi)  # Normalized for upright position

        # COMPONENT 1: Base reward for pendulum being upright (range: -1 to 1)
        # Uses cosine which is a naturally smooth function
        upright_reward = 2.0 * np.cos(alpha_norm)

        # COMPONENT 2: Smooth penalty for high velocities - quadratic falloff
        # Use tanh to create a smoother penalty that doesn't grow excessively large
        velocity_norm = (theta_dot ** 2 + alpha_dot ** 2) / 10.0  # Normalize velocities
        velocity_penalty = -0.3 * np.tanh(velocity_norm)  # Bounded penalty

        # COMPONENT 3: Smooth penalty for arm position away from center
        # Again using tanh for smooth bounded penalties
        pos_penalty = -0.1 * np.tanh(theta ** 2 / 2.0)

        # COMPONENT 4: Smoother bonus for being close to upright position
        upright_closeness = np.exp(-10.0 * alpha_norm ** 2)  # Close to 1 when near upright, falls off quickly
        stability_factor = np.exp(-1.0 * alpha_dot ** 2)  # Close to 1 when velocity is low
        bonus = 3.0 * upright_closeness * stability_factor  # Smoothly scales based on both factors

        # COMPONENT 4.5: Smoother cost for being close to downright position
        # For new convention, downright is at π
        downright_alpha = normalize_angle(alpha - np.pi)
        downright_closeness = np.exp(-10.0 * downright_alpha ** 2)
        stability_factor = np.exp(-1.0 * alpha_dot ** 2)
        bonus += -3.0 * downright_closeness * stability_factor  # Smoothly scales based on both factors

        # COMPONENT 5: Smoother penalty for approaching limits
        # Create a continuous penalty that increases as the arm approaches limits
        # Map the distance to limits to a 0-1 range, with 1 being at the limit
        theta_max_dist = np.clip(1.0 - abs(theta - THETA_MAX) / 0.5, 0, 1)
        theta_min_dist = np.clip(1.0 - abs(theta - THETA_MIN) / 0.5, 0, 1)
        limit_distance = max(theta_max_dist, theta_min_dist)

        # Apply a nonlinear function to create gradually increasing penalty
        # The penalty grows more rapidly as the arm gets very close to limits
        limit_penalty = -10.0 * limit_distance ** 3

        # COMPONENT 6: Energy management reward
        # This component is already quite smooth, just adjust scaling
        energy_reward = 2 - 0.15 * abs(Mp_g_Lp * (np.cos(alpha_norm))
                                       + 0.5 * Jp * alpha_dot ** 2
                                       - Mp_g_Lp)

        # Combine all components
        reward = (
                upright_reward
                # + velocity_penalty
                + pos_penalty
                + bonus
                + limit_penalty
                + energy_reward
        )

        return reward

    def render(self, mode='plot', save_path=None):
        """Visualize the pendulum trajectory

        Args:
            mode (str): Visualization mode ('plot' for matplotlib)
            save_path (str): Optional path to save the visualization
        """
        if mode == 'plot':
            states = np.array(self.state_history)
            actions = np.array(self.action_history) if self.action_history else np.zeros(len(states) - 1)

            plt.figure(figsize=(15, 12))

            # Plot state variables
            plt.subplot(3, 2, 1)
            plt.plot(states[:, 0])
            plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.3)
            plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.3)
            plt.title('Arm Angle (theta)')
            plt.xlabel('Step')
            plt.ylabel('Radians')

            plt.subplot(3, 2, 2)
            plt.plot(states[:, 1])
            plt.axhline(y=0, color='g', linestyle='--', alpha=0.3)  # Upright position
            plt.axhline(y=np.pi, color='r', linestyle='--', alpha=0.3)  # Downward position
            plt.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.3)  # Also downward position
            plt.title('Pendulum Angle (alpha)')
            plt.xlabel('Step')
            plt.ylabel('Radians')

            plt.subplot(3, 2, 3)
            plt.plot(states[:, 2])
            plt.title('Arm Angular Velocity')
            plt.xlabel('Step')
            plt.ylabel('Radians/s')

            plt.subplot(3, 2, 4)
            plt.plot(states[:, 3])
            plt.title('Pendulum Angular Velocity')
            plt.xlabel('Step')
            plt.ylabel('Radians/s')

            plt.subplot(3, 2, 5)
            plt.plot(self.reward_history)
            plt.title('Rewards per Step')
            plt.xlabel('Step')
            plt.ylabel('Reward')

            plt.subplot(3, 2, 6)
            plt.plot(actions)
            plt.axhline(y=max_voltage, color='r', linestyle='--', alpha=0.3)
            plt.axhline(y=-max_voltage, color='r', linestyle='--', alpha=0.3)
            plt.title('Control Actions (Voltage)')
            plt.xlabel('Step')
            plt.ylabel('Voltage')

            plt.tight_layout()

            # Save figure if path is provided
            if save_path:
                plt.savefig(save_path)

            plt.show()


def test_agent(agent, env, episodes=5):
    """Test the trained agent and visualize the results"""
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get action (using deterministic policy)
            action, _, _ = agent.get_action(state, deterministic=True)

            # Take step in environment
            state, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Test Episode {episode + 1}: Total Reward = {total_reward:.2f}")

        # Render the episode
        env.render()

    return total_reward


# Training function
def train_pendulum_ppo():
    """Train a PPO agent to control the pendulum system"""
    print("Starting PPO training for pendulum system...")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create the environment
    env = PendulumEnv(dt=0.01, max_steps=500)

    # Get dimensions from environment
    state_dim = env.observation_dim
    action_dim = 1  # Single motor voltage
    action_low = env.action_space_low
    action_high = env.action_space_high

    # Initialize PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        gamma=0.99,  # Discount factor
        clip_ratio=0.2,  # PPO clipping parameter
        policy_lr=3e-4,  # Learning rate for policy
        vf_lr=1e-3,  # Learning rate for value function
        train_iters=80,  # Number of policy updates per batch
        target_kl=0.01,  # Target KL divergence
        hidden_dim=64,  # Hidden dimension of neural networks
        lam=0.97  # GAE-Lambda parameter
    )

    # Train the agent
    rewards = agent.train(
        env=env,
        num_epochs=100,  # Number of epochs
        steps_per_epoch=4000,  # Steps per epoch
        batch_size=64,  # Batch size
        max_ep_len=500,  # Maximum episode length
        render_every=10  # Render every N epochs
    )

    print("Training complete!")

    # Test the trained agent
    test_agent(agent, env, episodes=5)

    return agent, rewards


# Execute the training if this script is run directly
if __name__ == "__main__":
    agent, rewards = train_pendulum_ppo()