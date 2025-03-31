import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb
from time import time

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

max_voltage = 8.0  # Maximum motor voltage
THETA_MIN = -2.2  # Minimum arm angle (radians)
THETA_MAX = 2.2  # Maximum arm angle (radians)

# Pre-compute constants for optimization
half_mL_LL_g = 0.5 * mL * LL * g
half_mL_LL_LA = 0.5 * mL * LL * LA
quarter_mL_LL_squared = 0.25 * mL * LL ** 2
Mp_g_Lp = mL * g * LL
Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia (kg·m²)


# Helper functions with numba optimization
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


# Simulation environment with RK4 integration like in SAC code
class PendulumEnv:
    def __init__(self, dt=0.014, max_steps=1300):
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
        # Convert normalized action to voltage
        voltage = float(action) * max_voltage if isinstance(action, np.ndarray) else action

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
                # + pos_penalty
                + bonus
                + limit_penalty
                + energy_reward
        )

        return reward


# Actor (Policy) Network - Keep the PPO architecture
class Actor(nn.Module):
    """Policy network for the PPO algorithm"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log_std heads for the action distribution
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Forward pass through the network"""
        x = self.network(state)
        mean = torch.tanh(self.mean(x))

        # Get the action distribution
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)

        return dist


# Critic (Value) Network - Keep the PPO architecture
class Critic(nn.Module):
    """Value network for the PPO algorithm"""

    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """Forward pass through the network"""
        return self.network(state)


# PPO Buffer - Modified to be more like SAC's buffer structure
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


# PPO Agent - Keeping the same functionality while matching SAC structure
class PPOAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            action_low=-1.0,
            action_high=1.0,
            gamma=0.99,
            clip_ratio=0.2,
            policy_lr=3e-4,
            vf_lr=3e-4,
            train_iters=80,
            target_kl=0.01,
            hidden_dim=256,
            lam=0.97
    ):
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
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)

        # For tracking training progress
        self.episode_rewards = []
        self.mean_rewards = []

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    def select_action(self, state, deterministic=False):
        """Get action from policy network (like SAC's select_action)"""
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state_tensor)

        if deterministic:
            action = dist.mean.detach().cpu().numpy()[0]
        else:
            action = dist.sample().detach().cpu().numpy()[0]

        # Clip action to environment's action space
        action = np.clip(action, self.action_low, self.action_high)

        # Get log probability of the action and value
        log_prob = dist.log_prob(torch.FloatTensor(action).to(self.device)).sum().detach().cpu().numpy()
        value = self.critic(state_tensor).detach().cpu().numpy()[0, 0]

        return action, log_prob, value

    def update_parameters(self, buffer):
        """Update the policy and value network using PPO with minibatches"""
        data = buffer.get()

        # Get full dataset
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device).unsqueeze(1)
        advantages = torch.FloatTensor(data['advantages']).to(self.device).unsqueeze(1)
        old_logp = torch.FloatTensor(data['logp_old']).to(self.device).unsqueeze(1)

        # Set up minibatch training
        batch_size = len(states)
        minibatch_size = 128  # Choose a reasonable size
        num_minibatches = batch_size // minibatch_size

        policy_losses = []
        value_losses = []
        kl_divs = []

        # Train for multiple epochs
        for _ in range(self.train_iters):
            # Generate random indices for minibatches
            indices = torch.randperm(batch_size)

            # Track KL divergence for early stopping
            total_kl = 0

            # Process minibatches
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_old_logp = old_logp[mb_idx]

                # Get current action distribution
                dist = self.actor(mb_states)
                values = self.critic(mb_states)

                # Calculate log probabilities of actions
                logp = dist.log_prob(mb_actions).sum(dim=1, keepdim=True)

                # Calculate ratio for PPO
                ratio = torch.exp(logp - mb_old_logp)

                # Calculate surrogate losses
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages

                # Policy loss (negative because we're maximizing)
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value function loss with clipping (PPO2-style)
                old_values = self.critic(mb_states).detach()
                values_clipped = old_values + torch.clamp(values - old_values, -self.clip_ratio, self.clip_ratio)
                v_loss1 = (values - mb_returns) ** 2
                v_loss2 = (values_clipped - mb_returns) ** 2
                value_loss = torch.max(v_loss1, v_loss2).mean()

                # Update the networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                policy_loss.backward()
                value_loss.backward()

                # Optional: Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.actor_scheduler.step()
                self.critic_scheduler.step()

                # Calculate approximate KL for early stopping
                approx_kl = ((mb_old_logp - logp) ** 2).mean().item()
                total_kl += approx_kl * (end - start) / batch_size

                # Track losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            # Early stopping based on KL divergence
            if total_kl > 1.5 * self.target_kl:
                break

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'kl_div': np.mean(kl_divs) if kl_divs else total_kl
        }

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


def train():
    """Training function, structured like the SAC training function"""
    print("Starting PPO training for inverted pendulum control...")

    # Environment setup
    env = PendulumEnv()

    # Hyperparameters
    state_dim = 6  # Our observation space
    action_dim = 1  # Motor voltage (normalized)
    max_episodes = 10000
    max_steps = 1300
    steps_per_epoch = 1300

    # Scale actions to motor voltage range
    action_low = -1.0  # Will be scaled to -max_voltage
    action_high = 1.0  # Will be scaled to max_voltage

    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        gamma=0.99,
        clip_ratio=0.2,
        policy_lr=3e-4,
        vf_lr=3e-4,
        train_iters=80,
        target_kl=0.01,
        hidden_dim=256,
        lam=0.97
    )

    # Initialize buffer for storing trajectories
    buffer = PPOBuffer(state_dim, action_dim, steps_per_epoch, gamma=0.99, lam=0.97)

    # Metrics
    episode_rewards = []
    avg_rewards = []

    # For visualization - store episode data
    states_history = []
    actions_history = []

    # Training loop
    start_time = time()

    total_steps = 0
    for epoch in range(max_episodes):
        epoch_start_time = time()

        # Collect trajectories
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        # If we're going to plot this episode, prepare to collect state data
        plot_this_episode = (epoch + 1) % 100 == 0
        if plot_this_episode:
            episode_states = []
            episode_actions = []

        for t in range(steps_per_epoch):
            # Get action
            action, log_prob, value = agent.select_action(state)

            # Scale action to motor voltage
            scaled_action = action * max_voltage

            # Take step in environment
            next_state, reward, done, _ = env.step(scaled_action)

            # Store transition in buffer
            buffer.store(state, action, reward, next_state, value, log_prob, done)

            # If we're plotting, store the raw state and action
            if plot_this_episode and episode_length < max_steps:
                episode_states.append(env.state.copy())
                episode_actions.append(action)

            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Handle episode termination or buffer full
            timeout = episode_length >= max_steps
            terminal = done or timeout

            if terminal or t == steps_per_epoch - 1:
                # If trajectory didn't reach terminal state, bootstrap value
                if not terminal:
                    _, _, last_val = agent.select_action(state)
                else:
                    last_val = 0

                buffer.finish_path(last_val)

                # Only record episode reward if it's a genuine episode termination
                if terminal:
                    episode_rewards.append(episode_reward)
                    last_completed_reward = episode_reward  # Store the last reward before resetting

                    # Reset for next episode
                state = env.reset()
                episode_reward = 0
                episode_length = 0

        # Update the policy and value networks after collecting a batch of data
        update_info = agent.update_parameters(buffer)

        # Calculate mean reward over last 10 episodes
        mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        avg_rewards.append(mean_reward)

        # Log progress
        if (epoch + 1) % 50 == 0:
            print(f"Episode {epoch + 1}/{max_episodes} | Reward: {last_completed_reward:.2f} | "
                  f"Avg Reward: {mean_reward:.2f} | Policy Loss: {update_info['policy_loss']:.4f} | "
                  f"Value Loss: {update_info['value_loss']:.4f} | KL Div: {update_info['kl_div']:.4f}")

            # Plot simulation for visual progress tracking
            if plot_this_episode:
                plot_training_episode(epoch, episode_states, episode_actions, env.dt, last_completed_reward)

            if (epoch + 1) % 100 == 0:
                # Save trained model
                timestamp = int(time())
                agent.save(f"{epoch + 1}_ppo_{timestamp}.pth")

        # Early stopping if well trained
        if mean_reward > 4900 and epoch > 250:
            print(f"Environment solved in {epoch + 1} episodes!")
            break

    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Save trained model
    timestamp = int(time())
    agent.save(f"ppo_{timestamp}.pth")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='10-Episode Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('PPO Training for Inverted Pendulum')
    plt.legend()
    plt.grid(True)
    plt.savefig("ppo_training_progress.png")
    plt.close()

    return agent


def plot_learning_curve(rewards, filename="ppo_learning_curve.png"):
    """Plot the learning curve showing episode rewards and moving average"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Episode Reward')

    # Calculate moving average with window size of 10
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        plt.plot(range(9, len(rewards)), moving_avg, 'r', label='10-Episode Moving Average')

    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('PPO Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def evaluate(agent, num_episodes=5, render=True):
    """Evaluate function, similar to the SAC evaluation function"""
    env = PendulumEnv()

    for episode in range(num_episodes):
        state = env.reset(random_init=False)  # Start from standard position
        total_reward = 0

        # Store episode data for plotting
        states_history = []
        actions_history = []

        for step in range(env.max_steps):
            # Select action (deterministic for evaluation)
            action, _, _ = agent.select_action(state, deterministic=True)

            # Scale action to motor voltage
            scaled_action = action * max_voltage

            # Step environment
            next_state, reward, done, _ = env.step(scaled_action)

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
            plt.title(f'PPO Inverted Pendulum Control - Episode {episode + 1}')
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
            plt.savefig(f"ppo_evaluation_episode_{episode + 1}.png")
            plt.close()

            print(f"Time spent balanced: {balanced_time:.2f} seconds")
            print(f"Data points with pendulum upright: {num_upright_points}")
            print(f"Max arm angle: {np.max(np.abs(thetas)):.2f} rad")
            print(f"Max pendulum angular velocity: {np.max(np.abs(alpha_dots)):.2f} rad/s")
            final_angle_deg = abs(alpha_normalized[-1]) * 180 / np.pi
            print(
                f"Final pendulum angle from vertical: {abs(alpha_normalized[-1]):.2f} rad ({final_angle_deg:.1f} degrees)")
            print("-" * 50)


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
    plt.scatter(t, alpha_normalized)
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
    plt.savefig(f"PPO_training_episode_{episode}.png")
    plt.close()  # Close the figure to avoid memory issues

    """print(f"Time spent balanced: {balanced_time:.2f} seconds")
    print(f"Data points with pendulum upright: {num_upright_points}")
    print(f"Max arm angle: {np.max(np.abs(thetas)):.2f} rad")
    print(f"Max pendulum angular velocity: {np.max(np.abs(alpha_dots)):.2f} rad/s")
    final_angle_deg = abs(alpha_normalized[-1]) * 180 / np.pi
    print(
        f"Final pendulum angle from vertical: {abs(alpha_normalized[-1]):.2f} rad ({final_angle_deg:.1f} degrees)")
    print("-" * 50)"""


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