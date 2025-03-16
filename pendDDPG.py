import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        self.action_space_low = -5.0  # Minimum voltage
        self.action_space_high = 5.0  # Maximum voltage
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

    def reset(self, curriculum_phase=0):
        """Reset the environment to initial state

        Args:
            curriculum_phase: Integer controlling the difficulty
                0 = Start hanging down (hardest)
                1 = Start with small random perturbation from upright (easier)
                2 = Start very close to upright (easiest)
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
        else:
            # Easy: Very close to upright
            theta = np.random.uniform(-0.1, 0.1)
            alpha = np.random.uniform(-0.2, 0.2)  # Within ~10° of upright
            theta_dot = 0
            alpha_dot = 0
            self.state = np.array([theta, alpha, theta_dot, alpha_dot])

        self.step_count = 0
        self.action_history = []  # Clear action history
        self.upright_counter = 0  # Reset upright counter
        return self.get_observation()

    def step(self, action):
        """Take a step in the environment"""
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


# Replay Buffer for DDPG
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        """Initialize a replay buffer for DDPG

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_counter = 0
        self.buffer_full = False

        # Storage for experiences
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer"""
        # Store experience at the current index
        idx = self.buffer_counter % self.capacity

        self.states[idx] = state
        # Ensure action is added as an array of the correct shape
        if isinstance(action, np.ndarray):
            self.actions[idx] = action.reshape(self.action_dim)
        else:
            self.actions[idx] = np.array([action], dtype=np.float32)
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        # Update counter and flag
        self.buffer_counter += 1
        if self.buffer_counter >= self.capacity:
            self.buffer_full = True

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        # Determine the actual size of the buffer
        max_idx = self.capacity if self.buffer_full else self.buffer_counter

        # Sample random indices
        indices = np.random.randint(0, max_idx, size=batch_size)

        # Return the sampled experiences
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def size(self):
        """Return the current size of the buffer"""
        return self.capacity if self.buffer_full else self.buffer_counter


# Actor Network for DDPG (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        """Initialize the Actor network

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum value for action
            hidden_dim: Dimension of hidden layers
        """
        super(Actor, self).__init__()

        self.max_action = max_action

        # Define the network architecture
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for better training dynamics"""
        for m in [self.layer1, self.layer2, self.layer3]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """Forward pass through the network"""
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        # Tanh activation to bound actions between -1 and 1, then scale to action range
        x = self.max_action * torch.tanh(self.layer3(x))
        return x


# Critic Network for DDPG (Q-Value)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """Initialize the Critic network

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super(Critic, self).__init__()

        # Define the network architecture
        # First layer processes only the state
        self.layer1_state = nn.Linear(state_dim, hidden_dim)

        # Second layer processes the combined state and action
        self.layer2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for better training dynamics"""
        for m in [self.layer1_state, self.layer2, self.layer3, self.layer4]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        """Forward pass through the network"""
        # Process state first
        x_state = F.relu(self.layer1_state(state))

        # Concatenate processed state with action
        x = torch.cat([x_state, action], dim=1)

        # Process the combined state-action input
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)

        return x


# DDPG Agent
class DDPGAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            hidden_dim=256,
            buffer_capacity=1000000,
            batch_size=64,
            actor_lr=1e-4,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,  # Target network update rate
            noise_std=0.1,  # Exploration noise standard deviation
            device="cpu"
    ):
        """Initialize DDPG agent

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            hidden_dim: Dimension of hidden layers
            buffer_capacity: Capacity of replay buffer
            batch_size: Batch size for updates
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            tau: Target network update rate
            noise_std: Standard deviation of exploration noise
            device: Computation device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.device = device

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

        # Initialize actor networks (policy)
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Initialize critic networks (Q-value)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state, add_noise=True):
        """Get an action from the policy with optional exploration noise"""
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()

        # Add exploration noise if requested (training mode)
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise

        # Clip action to valid range
        action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update(self):
        """Update the policy and value networks using a batch of experiences"""
        # Skip update if not enough samples
        if self.replay_buffer.size() < self.batch_size:
            return 0, 0  # Return zero losses if no update was performed

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ------------ Update Critic ------------
        # Get target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Get current Q-values
        current_q = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------ Update Actor ------------
        # Compute actor loss (maximize Q-value)
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------ Update Target Networks ------------
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def save_model(self, path):
        """Save model parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, path)

    def load_model(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


# Main training function with curriculum learning
def train_ddpg(env, agent, num_episodes=1000, max_steps_per_episode=5000, save_path='pendulum_ddpg_model.pt'):
    """Train the DDPG agent using curriculum learning"""
    episode_rewards = []
    best_avg_reward = -float('inf')

    # For tracking metrics
    actor_losses = []
    critic_losses = []

    # Curriculum learning phases
    curriculum_phases = [
        {"phase": 2, "episodes": int(num_episodes * 0.2)},  # Start with easiest (20% of episodes)
        {"phase": 1, "episodes": int(num_episodes * 0.3)},  # Then medium difficulty (30% of episodes)
        {"phase": 0, "episodes": int(num_episodes * 0.5)}  # Finally hardest setting (50% of episodes)
    ]

    current_phase = 0
    episode_in_phase = 0

    for episode in tqdm(range(num_episodes)):
        # Check if we should advance to the next curriculum phase
        if current_phase < len(curriculum_phases) - 1:
            if episode_in_phase >= curriculum_phases[current_phase]["episodes"]:
                current_phase += 1
                episode_in_phase = 0
                print(f"\nAdvancing to curriculum phase {current_phase + 1}")

        # Current curriculum phase
        curriculum_phase = curriculum_phases[current_phase]["phase"]
        episode_in_phase += 1

        # Reset environment
        state = env.reset(curriculum_phase=curriculum_phase)
        episode_reward = 0

        # Run one episode
        for step in range(max_steps_per_episode):
            # Get action from agent (with exploration noise during training)
            action = agent.get_action(state, add_noise=True)

            # Take step in environment
            next_state, reward, done, _ = env.step(action)

            # Store experience in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, float(done))

            # Update state and accumulate reward
            state = next_state
            episode_reward += reward

            # Update the networks
            if step % 2 == 0:  # Update every 2 steps to gather more experience
                actor_loss, critic_loss = agent.update()
                if actor_loss != 0:  # Only append non-zero losses
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

            if done:
                break

        # Record episode reward
        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Phase: {curriculum_phase}")

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(save_path)
                print(f"New best model saved with avg reward: {best_avg_reward:.2f}")

        # Visualize periodically
        if (episode + 1) % 100 == 0 or episode == num_episodes - 1:
            visualize_policy(env, agent)

    # Plot training progress
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    if actor_losses:  # Only plot if we have data
        plt.subplot(1, 3, 2)
        plt.plot(actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('ddpg_training_progress.png')
    plt.show()

    return episode_rewards


# Function to visualize the trained policy
def visualize_policy(env, agent, max_steps=5_000):
    """Visualize the trained policy"""
    # Initialize arrays for storing trajectories
    t_array = np.arange(0, max_steps * env.dt, env.dt)
    theta_array = np.zeros(max_steps)
    alpha_array = np.zeros(max_steps)
    action_array = np.zeros(max_steps)
    reward_array = np.zeros(max_steps)

    # Run an episode with the trained policy
    state = env.reset(curriculum_phase=0)  # Start from hardest setting (hanging down)
    for i in range(max_steps):
        # Get action from agent (deterministic mode)
        action = agent.get_action(state, add_noise=False)

        # Store current state and action
        theta_array[i] = env.state[0]  # Access the actual theta value from environment
        alpha_array[i] = env.state[1]  # Access the actual alpha value from environment
        # Ensure action is a scalar if it's wrapped in an array
        action_array[i] = action[0] if isinstance(action, np.ndarray) and action.size == 1 else action

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
    plt.title('DDPG Controller Performance')
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
    plt.savefig('ddpg_performance.png')
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
        states.append(env.state.copy())  # Store the actual state values

        # Get action from policy (deterministic mode)
        action = agent.get_action(state, add_noise=False)
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
        flat_actions = np.array([a.item() if hasattr(a, 'item') else a for a in actions[:i + 1]])
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
    plt.suptitle('QUBE-Servo 2 Pendulum with DDPG Control', fontsize=16)
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
def main():
    # Create environment
    env = PendulumEnv(dt=0.001, max_steps=10_000)

    # Define agent parameters
    state_dim = env.observation_space_dim
    action_dim = 1  # Single control input (voltage)
    max_action = env.action_space_high  # Maximum voltage

    # Create DDPG agent
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=256,  # Larger network capacity
        buffer_capacity=100000,  # Large replay buffer
        batch_size=64,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,  # Discount factor
        tau=0.005,  # Soft target update rate
        noise_std=0.2,  # Exploration noise
        device=device
    )

    # Train the agent
    print("Training the agent...")
    train_ddpg(env, agent, num_episodes=50, max_steps_per_episode=5000)

    # Visualize the trained policy
    print("Visualizing trained policy...")
    visualize_policy(env, agent)

    # Animate the final policy
    print("Animating pendulum control...")
    animate_pendulum_episode(env, agent, save_path='ddpg_pendulum_control.gif')


if __name__ == "__main__":
    main()