import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Device configuration - set to 'cuda' for GPU, 'cpu' for CPU, or 'auto' to use GPU if available
DEVICE_TYPE = 'cpu'  # Options: 'cuda', 'cpu', 'auto'
# Actually faster with CPU

# Device setup based on configuration
if DEVICE_TYPE == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif DEVICE_TYPE == 'cuda' and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Helper function to normalize angles to [-π, π]
def angle_normalize(x):
    """Normalize angle to [-π, π] range"""
    if isinstance(x, torch.Tensor):
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi
    else:
        return ((x + np.pi) % (2 * np.pi)) - np.pi


# Define the RL environment wrapper for the pendulum simulation
class PendulumEnv:
    def __init__(self, dt=0.001, max_steps=10000):
        # System Parameters (from the QUBE-Servo 2 manual)
        # Motor and Pendulum parameters
        self.Rm = 8.4  # Motor resistance (Ohm)
        self.kt = 0.042  # Motor torque constant (N·m/A)
        self.km = 0.042  # Motor back-EMF constant (V·s/rad)
        self.Jm = 4e-6  # Motor moment of inertia (kg·m²)
        self.mh = 0.016  # Hub mass (kg)
        self.rh = 0.0111  # Hub radius (m)
        self.Jh = 0.6e-6  # Hub moment of inertia (kg·m^2)
        self.Mr = 0.095  # Rotary arm mass (kg)
        self.Lr = 0.085  # Arm length, pivot to end (m)
        self.Mp = 0.024  # Pendulum mass (kg)
        self.Lp = 0.129  # Pendulum length from pivot to center of mass (m)
        self.Jp = (1 / 3) * self.Mp * self.Lp ** 2  # Pendulum moment of inertia (kg·m²)
        self.Br = 0.001  # Rotary arm viscous damping coefficient (N·m·s/rad)
        self.Bp = 0.0001  # Pendulum viscous damping coefficient (N·m·s/rad)
        self.g = 9.81  # Gravity constant (m/s²)
        self.Jr = self.Jm + self.Jh + self.Mr * self.Lr ** 2 / 3  # Arm inertia

        # Simulation parameters
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0

        # For tracking action history (for variance penalty)
        self.action_history = []
        self.history_length = 10  # Number of actions to keep in history

        # Action and observation spaces
        self.action_space_low = -18.0  # Minimum voltage
        self.action_space_high = 18.0  # Maximum voltage
        self.observation_space_dim = 6  # [cos(θ), sin(θ), cos(α), sin(α), θ_dot, α_dot]

        # Current state
        self.state = np.zeros(4)

        # Initialize success tracking
        self.upright_counter = 0

    def get_observation(self):
        """Convert internal state to observation with trigonometric features"""
        theta, alpha, theta_dot, alpha_dot = self.state

        # Convert angles to their trigonometric representation
        obs = np.array([
            np.cos(theta),  # cos(theta)
            np.sin(theta),  # sin(theta)
            np.cos(alpha),  # cos(alpha)
            np.sin(alpha),  # sin(alpha)
            theta_dot,  # theta_dot (unchanged)
            alpha_dot  # alpha_dot (unchanged)
        ])

        return obs

    def set_time_limit(self, seconds):
        """Set a time limit for the episode"""
        self.max_episode_time = seconds
        self.episode_start_time = time.time()

    def reset(self, curriculum_phase=0):
        """Reset the environment to initial state

        Args:
            curriculum_phase: Integer controlling the difficulty
                0 = Start hanging down (hardest)
                1 = Start with small random perturbation from upright (easier)
                2 = Start very close to upright (easiest)
                3 = Start in a completely random state (unpredictable)
        """
        if curriculum_phase == 0:
            # Standard reset: pendulum hanging down
            self.state = np.array([0, np.pi, 0, 0])
        elif curriculum_phase == 1:
            # Medium difficulty: Small random perturbation from upright
            theta = np.random.uniform(-0.3, 0.3)
            alpha = np.random.uniform(-0.5, 0.5)  # Within ~30° of upright
            theta_dot = np.random.uniform(-0.1, 0.1)
            alpha_dot = np.random.uniform(-0.1, 0.1)
            self.state = np.array([theta, alpha, theta_dot, alpha_dot])
        elif curriculum_phase == 2:
            # Easy: Very close to upright
            theta = np.random.uniform(-0.1, 0.1)
            alpha = np.random.uniform(-0.2, 0.2)  # Within ~10° of upright
            theta_dot = 0
            alpha_dot = 0
            self.state = np.array([theta, alpha, theta_dot, alpha_dot])
        elif curriculum_phase == 3:
            # Completely random state: full range of possible positions and velocities
            theta = np.random.uniform(-np.pi, np.pi)  # Random arm angle in full circle
            alpha = np.random.uniform(-np.pi, np.pi)  # Random pendulum angle in full circle
            theta_dot = np.random.uniform(-2.0, 2.0)  # Random arm angular velocity
            alpha_dot = np.random.uniform(-2.0, 2.0)  # Random pendulum angular velocity
            self.state = np.array([theta, alpha, theta_dot, alpha_dot])
        else:
            # Fallback to default (hanging down) if invalid curriculum phase
            self.state = np.array([0, np.pi, 0, 0])
            print(f"Warning: Invalid curriculum_phase {curriculum_phase}. Using default (0).")

        self.step_count = 0
        self.action_history = []  # Clear action history
        self.upright_counter = 0  # Reset upright counter

        # Reset the episode timer if we're using one
        if hasattr(self, 'max_episode_time'):
            self.episode_start_time = time.time()

        return self.get_observation()

    def step(self, action):
        """Take a step in the environment with optional time tracking"""
        # If we're tracking episode time, check if we've exceeded the limit
        if hasattr(self, 'episode_start_time') and hasattr(self, 'max_episode_time'):
            elapsed = time.time() - self.episode_start_time
            if elapsed > self.max_episode_time:
                # Return done=True if time limit is reached
                return self.get_observation(), 0, True, {"timeout": True, "is_upright": False}

        # Ensure action is within bounds
        action = np.clip(action, self.action_space_low, self.action_space_high)

        # Convert action to scalar if it's an array
        if isinstance(action, np.ndarray):
            action = float(action.item())  # Extract the scalar value

        # Add action to history for variance calculation
        self.action_history.append(action)
        if len(self.action_history) > self.history_length:
            self.action_history.pop(0)  # Remove oldest action

        # Apply dynamics
        derivatives = self.pendulum_dynamics(self.state, action)
        self.state = self.state + self.dt * derivatives

        # Normalize angles to [-π, π]
        self.state[0] = angle_normalize(self.state[0])
        self.state[1] = angle_normalize(self.state[1])

        # Calculate reward
        reward = self.calculate_reward(action)

        # Check if pendulum is upright (for early termination)
        is_upright = abs(angle_normalize(self.state[1])) < 0.2  # Within ~11 degrees of upright
        if is_upright:
            self.upright_counter += 1
        else:
            self.upright_counter = 0

        # Success bonus: if stayed upright for sufficient time
        success_bonus = 0
        if self.upright_counter >= 500:  # 0.5 seconds at dt=0.001
            success_bonus = 50

        # Check if episode is done
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # Early termination with bonus if successfully balanced
        if self.upright_counter >= 1000:  # 1 second upright
            done = True
            success_bonus = 100  # Larger bonus for maintaining upright position

        # Return state, reward with any bonus, done flag, and info
        return self.get_observation(), reward + success_bonus, done, {"is_upright": is_upright}

    def calculate_reward(self, action):
        """Calculate reward based on pendulum state"""
        theta, alpha, theta_dot, alpha_dot = self.state

        # Normalize alpha to [-π, π] range for reward calculation
        alpha_normalized = angle_normalize(alpha)

        # Reward components:

        # 1. Main upright position reward: higher when closer to upright
        # Using squared angle gives smoother gradient and higher penalty as angle increases
        upright_reward = -1.0 * (alpha_normalized ** 2)

        # 2. Cosine component provides additional signal and is maximized at upright
        cos_reward = 2.0 * np.cos(alpha_normalized)

        # 3. Penalize high angular velocities (for stability)
        velocity_penalty = -0.1 * (theta_dot ** 2 + alpha_dot ** 2)

        # 4. Small penalty for arm angle away from center
        arm_penalty = -0.1 * theta ** 2

        # 5. Control effort penalty (penalize large actions)
        action_magnitude_penalty = -0.01 * action ** 2

        # 6. Action variance penalty (if we have enough history)
        action_variance_penalty = 0
        if len(self.action_history) > 1:
            action_variance = np.var(self.action_history)
            action_variance_penalty = -0.1 * action_variance

        # Combine rewards
        reward = upright_reward + cos_reward + velocity_penalty + arm_penalty + action_magnitude_penalty + action_variance_penalty

        return reward

    def pendulum_dynamics(self, state, vm):
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
        im = (vm - self.km * theta_dot) / self.Rm

        # Motor torque
        tau = self.kt * im

        # Equations of motion
        # Inertia matrix elements
        M11 = self.Jr + self.Mp * self.Lr ** 2
        M12 = self.Mp * self.Lr * self.Lp / 2 * np.cos(alpha)
        M21 = M12
        M22 = self.Jp

        # Coriolis and centrifugal terms
        C1 = -self.Mp * self.Lr * (self.Lp / 2) * alpha_dot ** 2 * np.sin(alpha) - self.Br * theta_dot
        C2 = self.Mp * self.g * (self.Lp / 2) * np.sin(alpha) - self.Bp * alpha_dot

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


# Neural network for the policy (actor)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        # Deeper network with better initialization
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Tanh to bound actions between -1 and 1
        )

        # Initialize with appropriate scaling for better gradient flow
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Log standard deviation with better initial value for exploration
        # Starting with higher std dev (-1.0 instead of 0.0) encourages more exploration
        self.log_std = nn.Parameter(torch.ones(output_dim) * -1.0)

    def forward(self, state):
        # Mean of the action distribution
        action_mean = 5.0 * self.actor(state)  # Scale output to match action space

        # Standard deviation with clipping for stability
        action_std = torch.exp(torch.clamp(self.log_std, -20, 2))

        # Create normal distribution
        dist = Normal(action_mean, action_std)

        return dist


# Neural network for the value function (critic)
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()

        # Deeper network for value function
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize with appropriate scaling
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.critic(state)


# PPO Agent
class PPOAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=128, lr=3e-4, gamma=0.99,
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, device="cpu"):
        # Store the device
        self.device = device

        # Initialize the policy and value networks
        self.policy = PolicyNetwork(input_dim, output_dim, hidden_dim).to(device)
        self.value = ValueNetwork(input_dim, hidden_dim).to(device)

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def get_action(self, state, training=True):
        """Sample an action from the policy distribution"""
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            dist = self.policy(state_tensor)

            if training:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                # Keep log_prob as array for batch updates
                return action.cpu().numpy(), log_prob.cpu().numpy()
            else:
                # During testing, just use the mean action
                action = dist.mean
                return action.cpu().numpy(), None

    def update(self, states, actions, advantages, returns, old_log_probs, epochs=10, batch_size=64):
        """Update the policy and value networks using PPO"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        # Normalize advantages (helps with training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-batch updates
        for _ in range(epochs):
            # Generate random indices
            indices = np.random.permutation(len(states))

            # Mini-batch training
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch
                idx = indices[start_idx:start_idx + batch_size]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]

                # Get current policy distribution and values
                dist = self.policy(batch_states)
                values = self.value(batch_states).squeeze()

                # Calculate log probabilities of actions
                log_probs = dist.log_prob(batch_actions)

                # Calculate ratio for PPO update
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # PPO objective function
                obj = ratio * batch_advantages
                clipped_obj = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(obj, clipped_obj).mean()

                # Calculate value loss (MSE)
                value_loss = ((values - batch_returns) ** 2).mean()

                # Calculate entropy for exploration
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

    def save_model(self, path):
        """Save model parameters"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, path)

    def load_model(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])


# Function to compute returns and advantages
def compute_advantages_and_returns(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute advantages and returns using GAE (Generalized Advantage Estimation)"""
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = 0.0  # End of episode
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns


# Main training function with curriculum learning
def train_ppo(env, agent, num_episodes=1000, steps_per_update=2000, save_path='pendulum_ppo_model.pt', max_time=None):
    """Train the PPO agent using curriculum learning"""
    import time
    start_time = time.time()

    episode_rewards = []
    best_avg_reward = -float('inf')

    # Curriculum learning phases
    curriculum_phases = [
        {"phase": 3, "episodes": int(num_episodes * 0.2)},  #2 Start with easiest (20% of episodes)
        {"phase": 3, "episodes": int(num_episodes * 0.3)},  #1 Then medium difficulty (30% of episodes)
        {"phase": 3, "episodes": int(num_episodes * 0.5)}  #0 Finally hardest setting (50% of episodes)
    ]

    current_phase = 0
    episode_in_phase = 0

    for episode in tqdm(range(num_episodes)):
        # Check time limit if specified
        if max_time is not None and (time.time() - start_time) > max_time:
            print(f"\nReached time limit of {max_time} seconds. Stopping training.")
            break

        # Check if we should advance to the next curriculum phase
        if current_phase < len(curriculum_phases) - 1:
            if episode_in_phase >= curriculum_phases[current_phase]["episodes"]:
                current_phase += 1
                episode_in_phase = 0
                print(f"\nAdvancing to curriculum phase {current_phase + 1}")

        # Current curriculum phase
        curriculum_phase = curriculum_phases[current_phase]["phase"]
        episode_in_phase += 1

        # Storage for batch updates
        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_dones = []
        all_values = []

        step_count = 0
        episode_reward = 0

        while step_count < steps_per_update:
            state = env.reset(curriculum_phase=curriculum_phase)
            done = False
            episode_step = 0

            while not done and step_count < steps_per_update:
                # Get action from agent
                action, log_prob = agent.get_action(state)

                # Get value estimate
                value = agent.value(torch.FloatTensor(state).to(device)).item()

                # Take step in environment
                next_state, reward, done, info = env.step(action)

                # Store data
                all_states.append(state)
                all_actions.append(action)
                all_log_probs.append(log_prob)
                all_rewards.append(reward)
                all_dones.append(done)
                all_values.append(value)

                # Update state
                state = next_state
                episode_reward += reward
                step_count += 1
                episode_step += 1

                if done:
                    break

        # Convert to numpy arrays
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_log_probs = np.array(all_log_probs)
        all_rewards = np.array(all_rewards)
        all_dones = np.array(all_dones)
        all_values = np.array(all_values)

        # Compute advantages and returns
        advantages, returns = compute_advantages_and_returns(all_rewards, all_values, all_dones, agent.gamma)

        # Update the agent
        agent.update(all_states, all_actions, advantages, returns, all_log_probs)

        # Record episode reward
        episode_rewards.append(episode_reward)

        # Print time elapsed with progress reports
        if (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Phase: {curriculum_phase}, Time: {elapsed_time:.1f}s")

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(save_path)
                print(f"New best model saved with avg reward: {best_avg_reward:.2f}")

            # Visualize periodically
            if (episode + 1) % 100 == 0 or episode == num_episodes - 1:
                visualize_policy(env, agent)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")



    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()

    return episode_rewards


# Function to visualize the trained policy
def visualize_policy(env, agent, max_steps=5_000, max_time=None):
    """Visualize the trained policy"""
    start_time = time.time()
    # Initialize arrays for storing trajectories
    t_array = np.arange(0, max_steps * env.dt, env.dt)
    theta_array = np.zeros(max_steps)
    alpha_array = np.zeros(max_steps)
    action_array = np.zeros(max_steps)
    reward_array = np.zeros(max_steps)

    # Run an episode with the trained policy
    state = env.reset(curriculum_phase=0)  # Start from hardest setting (hanging down)
    for i in range(max_steps):
        # Check time limit
        if max_time is not None and (time.time() - start_time) > max_time:
            print(f"Reached time limit of {max_time} seconds")
            break

        # Get action from agent (deterministic mode)
        action, _ = agent.get_action(state, training=False)

        # Store current state and action
        theta_array[i] = state[0]
        alpha_array[i] = state[1]
        # Ensure action is a scalar
        action_array[i] = float(action.item()) if isinstance(action, np.ndarray) else action

        # Take step in environment and record reward
        next_state, reward, done, _ = env.step(action)
        reward_array[i] = reward

        # Update state
        state = next_state

        if done:
            break

    # Trim arrays to actual length
    t_array = t_array[:i + 1]
    theta_array = theta_array[:i + 1]
    alpha_array = alpha_array[:i + 1]
    action_array = action_array[:i + 1]
    reward_array = reward_array[:i + 1]

    # Plot the results
    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    plt.plot(t_array, action_array)
    plt.ylabel('Input Voltage (V)')
    plt.title('RL Controller Performance')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(t_array, theta_array, label='Arm angle (θ)')
    plt.ylabel('Arm Angle (rad)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(t_array, alpha_array, label='Pendulum angle (α)')
    plt.ylabel('Pendulum Angle (rad)')
    plt.legend()
    plt.grid(True)

    # Highlight the upright position
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.subplot(4, 1, 4)
    plt.plot(t_array, reward_array)
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rl_performance.png')
    plt.show()

    # Print summary statistics
    print(f"Episode completed in {i + 1} steps ({(i + 1) * env.dt:.2f} seconds)")
    print(f"Final pendulum angle: {alpha_array[-1]:.2f} rad ({np.degrees(alpha_array[-1]):.2f} degrees)")
    print(f"Total reward: {np.sum(reward_array):.2f}")

    # Check if pendulum was successfully inverted
    success = abs(angle_normalize(alpha_array[-1])) < 0.2
    print(f"Inversion successful: {success}")

    return t_array, theta_array, alpha_array, action_array, reward_array


# Run an example episode with animation that includes angle plots
def animate_pendulum_episode(env, agent, save_path=None, max_frames=500):
    """Run and animate a complete episode with the trained agent, including plots of alpha and theta"""
    from matplotlib import animation
    import matplotlib.pyplot as plt

    # Run the policy and collect state history
    max_steps = 5000
    state = env.reset(curriculum_phase=0)  # Start hanging down (hardest)

    # Arrays to store trajectory
    states = []
    actions = []

    print("Simulating pendulum trajectory...")
    for i in range(max_steps):
        # Store current state
        states.append(state.copy())

        # Get action from policy
        action, _ = agent.get_action(state, training=False)
        actions.append(action)

        # Take step in environment
        state, _, done, _ = env.step(action)

        # Optional early stopping if successfully inverted
        if done:
            print(f"Episode complete after {i + 1} steps")
            break

    # Downsample for animation (to prevent memory issues)
    print(f"Creating animation with {len(states)} states...")
    if len(states) > max_frames:
        # Take evenly spaced frames
        indices = np.linspace(0, len(states) - 1, max_frames, dtype=int)
        states = [states[i] for i in indices]
        actions = [actions[i] for i in indices]

    # Convert to array
    states = np.array(states)
    actions = np.array(actions)
    print(f"Animation will have {len(states)} frames")

    # Create animation with multiple subplots
    fig = plt.figure(figsize=(12, 10))

    # Create a layout with 4 subplots
    gs = fig.add_gridspec(3, 2)

    # Large pendulum visualization
    ax_pendulum = fig.add_subplot(gs[0:2, 0], aspect='equal', autoscale_on=False, xlim=(-0.2, 0.2), ylim=(-0.2, 0.2))
    ax_pendulum.grid()

    # Angle plots
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_alpha = fig.add_subplot(gs[1, 1])
    ax_action = fig.add_subplot(gs[2, :])

    # Set titles and labels
    ax_pendulum.set_title('Pendulum Animation')
    ax_theta.set_title('Arm Angle (θ)')
    ax_alpha.set_title('Pendulum Angle (α)')
    ax_action.set_title('Control Action (Voltage)')

    ax_theta.set_ylabel('Angle (rad)')
    ax_alpha.set_ylabel('Angle (rad)')
    ax_action.set_ylabel('Voltage (V)')
    ax_action.set_xlabel('Time (s)')

    # Set up grid for all plots
    ax_theta.grid(True)
    ax_alpha.grid(True)
    ax_action.grid(True)

    # Elements to be animated
    pendulum_line, = ax_pendulum.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax_pendulum.text(0.05, 0.9, '', transform=ax_pendulum.transAxes)

    # Initialize the plot lines
    max_time = len(states) * env.dt * (max_steps / len(states))
    time_data = np.linspace(0, max_time, len(states))

    # Initialize static x axis for all plots
    ax_theta.set_xlim(0, max_time)
    ax_alpha.set_xlim(0, max_time)
    ax_action.set_xlim(0, max_time)

    # Set reasonable y limits based on expected angle ranges
    ax_theta.set_ylim(-np.pi, np.pi)
    ax_alpha.set_ylim(-np.pi, np.pi)
    action_max = max(1.0, np.max(np.abs(actions)) * 1.1)  # 10% margin
    ax_action.set_ylim(-action_max, action_max)

    # Add horizontal line at 0 for reference
    ax_theta.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax_alpha.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax_action.axhline(y=0, color='r', linestyle='--', alpha=0.3)

    # Create empty plot lines for the angle and action data
    theta_line, = ax_theta.plot([], [], 'b-')
    alpha_line, = ax_alpha.plot([], [], 'g-')
    action_line, = ax_action.plot([], [], 'r-')

    # Vertical time indicators for all plots
    theta_time_line = ax_theta.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    alpha_time_line = ax_alpha.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    action_time_line = ax_action.axvline(x=0, color='k', linestyle='-', alpha=0.5)

    # Physical dimensions
    arm_length = 0.085  # Rotary arm length (m)
    pend_length = 0.129  # Full pendulum length (m)

    def init():
        pendulum_line.set_data([], [])
        time_text.set_text('')
        theta_line.set_data([], [])
        alpha_line.set_data([], [])
        action_line.set_data([], [])
        # Vertical lines need sequences, not single values
        theta_time_line.set_xdata([0])
        alpha_time_line.set_xdata([0])
        action_time_line.set_xdata([0])
        return pendulum_line, time_text, theta_line, alpha_line, action_line, theta_time_line, alpha_time_line, action_time_line

    def animate(i):
        # Current time
        current_time = time_data[i]

        # Extract state
        theta = states[i, 0]
        alpha = states[i, 1]

        # Rotary arm endpoint
        arm_x = arm_length * np.cos(theta)
        arm_y = arm_length * np.sin(theta)

        # Pendulum endpoint
        pend_x = arm_x + pend_length * np.sin(alpha)
        pend_y = arm_y - pend_length * np.cos(alpha)

        # Update pendulum line data
        pendulum_line.set_data([0, arm_x, pend_x], [0, arm_y, pend_y])
        time_text.set_text(time_template % current_time)

        # Update angle plots up to current time
        theta_line.set_data(time_data[:i + 1], states[:i + 1, 0])
        alpha_line.set_data(time_data[:i + 1], states[:i + 1, 1])

        # Flatten action if it's a multi-dimensional array
        flat_actions = np.array([a.item() if isinstance(a, np.ndarray) else a for a in actions[:i + 1]])
        action_line.set_data(time_data[:i + 1], flat_actions)

        # Update time indicator lines (must use lists for vertical lines)
        theta_time_line.set_xdata([current_time])
        alpha_time_line.set_xdata([current_time])
        action_time_line.set_xdata([current_time])

        return pendulum_line, time_text, theta_line, alpha_line, action_line, theta_time_line, alpha_time_line, action_time_line

    print("Creating animation...")
    ani = animation.FuncAnimation(fig, animate, frames=len(states),
                                  interval=50, blit=True, init_func=init)

    plt.tight_layout()
    plt.suptitle('QUBE-Servo 2 Pendulum with RL Control', fontsize=16)
    plt.subplots_adjust(top=0.92)  # Adjust to make room for the suptitle

    # Save animation if path is provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        try:
            # Try to use a more efficient writer
            writer = animation.PillowWriter(fps=30)
            ani.save(save_path, writer=writer)
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Displaying animation without saving...")

    print("Displaying animation...")
    plt.show()
    print("Animation complete")

    return ani


# Main function to run the training and visualization
def main(train_time_limit=None, viz_time_limit=None, episode_time_limit=None):
    # Create environment and agent
    env = PendulumEnv(dt=0.001, max_steps=10_000)

    # Set episode time limit if specified
    if episode_time_limit is not None:
        env.set_time_limit(episode_time_limit)

    agent = PPOAgent(
        input_dim=env.observation_space_dim,
        output_dim=1,  # Single action (voltage)
        hidden_dim=128,  # Larger network capacity
        lr=3e-4,
        gamma=0.99,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.05,  # Increased exploration
        device = device  # Pass the device
    )

    # Train the agent with time limit
    print("Training the agent...")
    train_ppo(env, agent, num_episodes=20_000, steps_per_update=2000, max_time=train_time_limit)

    # Visualize the trained policy with time limit
    print("Visualizing trained policy...")
    visualize_policy(env, agent, max_time=viz_time_limit)

    # Animate the final policy
    print("Animating pendulum control...")
    animate_pendulum_episode(env, agent, save_path='pendulum_control.gif')


if __name__ == "__main__":
    # Train for at most 1 hour, limit visualization to 5 minutes, limit each episode to 30 seconds
    main(train_time_limit=8*60*60, viz_time_limit=300, episode_time_limit=30)