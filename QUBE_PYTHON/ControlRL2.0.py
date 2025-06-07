import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog, messagebox, Checkbutton, IntVar, StringVar, \
    OptionMenu
from QUBE import QUBE
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import csv
import threading
from datetime import datetime

# Try to import from SimRL for code reuse
try:
    from SimRL import ReplayBuffer, Critic, SACAgent

    print("Successfully imported components from SimRL")
except ImportError:
    print("Could not import from SimRL. Using built-in implementations.")
    # Define here if imports fail (implementations below)

# Update with your COM port
COM_PORT = "COM3"


# Low Pass Filter class (similar to the Arduino implementation)
class LowPassFilter:
    def __init__(self, cutoff_freq=500.0):
        """Initialize a low-pass filter with specified cutoff frequency in Hz"""
        self.twopi = 2.0 * np.pi
        self.wc = cutoff_freq / self.twopi  # Cutoff frequency parameter
        self.y_last = 0.0  # Last output value

    def filter(self, x, dt):
        """Apply filter to input x with timestep dt"""
        # Same equation as in the Arduino code:
        # y_k = y_k_last + wc*dt*(x - y_k_last)
        y_k = self.y_last + self.wc * dt * (x - self.y_last)
        self.y_last = y_k  # Save for next iteration
        return y_k

    def reset(self, initial_value=0.0):
        """Reset the filter state"""
        self.y_last = initial_value


# Actor Network
class Actor(nn.Module):
    """Policy network that outputs action distribution."""

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
        """Sample action from the distribution and compute log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from normal distribution with reparameterization trick
        normal = Normal(mean, std)
        x = normal.rsample()

        # Constrain to [-1, 1]
        action = torch.tanh(x)

        # Calculate log probability with squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


# Critic Network - only include if import from SimRL fails
if 'Critic' not in globals():
    class Critic(nn.Module):
        """Value network that estimates Q-values."""

        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super(Critic, self).__init__()

            # Q1 network
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

            # Q2 network to reducing overestimation bias
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, state, action):
            """Forward pass through both Q networks."""
            x = torch.cat([state, action], 1)

            q1 = self.q1(x)
            q2 = self.q2(x)

            return q1, q2

# ReplayBuffer - only include if import from SimRL fails
if 'ReplayBuffer' not in globals():
    class ReplayBuffer:
        """Store and sample transitions for off-policy learning."""

        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = []
            self.position = 0

        def push(self, state, action, reward, next_state, done):
            """Add a transition to the buffer."""
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            """Randomly sample a batch of transitions."""
            batch = np.random.choice(len(self.buffer), batch_size, replace=False)
            states, actions, rewards, next_states, dones = map(np.array, zip(*[self.buffer[i] for i in batch]))
            return states, actions, rewards, next_states, dones

        def __len__(self):
            return len(self.buffer)

# SACAgent - only include if import from SimRL fails
if 'SACAgent' not in globals():
    class SACAgent:
        """Soft Actor-Critic agent for continuous control."""

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
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

            # Automatic entropy tuning
            if automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
                self.log_alpha = torch.zeros(1, requires_grad=True)
                self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Move networks to device
            self.actor.to(self.device)
            self.critic.to(self.device)
            self.critic_target.to(self.device)

            if automatic_entropy_tuning:
                self.log_alpha = self.log_alpha.to(self.device)

        def select_action(self, state, evaluate=False):
            """Select an action given a state."""
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

            with torch.no_grad():
                if evaluate:
                    # Use mean action (no exploration) for evaluation
                    action, _ = self.actor(state)
                else:
                    # Sample action for training (with exploration)
                    action, _ = self.actor.sample(state)

                return action.cpu().numpy()[0]

        def update_parameters(self, memory, batch_size=512):
            """Update actor and critic parameters using a batch of experiences."""
            # Sample batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

            # Convert to tensors
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

            # === Update Critic ===
            with torch.no_grad():
                # Get next actions and log probs from current policy
                next_action, next_log_prob = self.actor.sample(next_state_batch)

                # Get target Q values
                target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
                target_q = torch.min(target_q1, target_q2)

                # Apply entropy term
                if self.automatic_entropy_tuning:
                    alpha = self.log_alpha.exp()
                else:
                    alpha = self.alpha

                # Compute TD target
                target_q = target_q - alpha * next_log_prob
                target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

            # Current Q estimates
            current_q1, current_q2 = self.critic(state_batch, action_batch)

            # Compute critic loss (MSE)
            critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # === Update Actor ===
            # Sample actions and log probs from current policy
            actions, log_probs = self.actor.sample(state_batch)

            # Get Q values for new actions
            q1, q2 = self.critic(state_batch, actions)
            min_q = torch.min(q1, q2)

            # Get current alpha
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

            # === Update Alpha (if automatic entropy tuning) ===
            if self.automatic_entropy_tuning:
                # Alpha loss
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                # Optimize alpha
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

            # === Soft Update Target Networks ===
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha': self.alpha.item() if self.automatic_entropy_tuning else self.alpha
            }


# Helper functions
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


class QUBEControllerWithRL:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        master.title("QUBE Controller with Reinforcement Learning Training")

        # Control state
        self.motor_voltage = 0.0
        self.target_position = 0.0
        self.calibrating = False
        self.rl_mode = False
        self.moving_to_position = False
        self.rl_model = None
        self.system_max_voltage = 18.0  # Increased maximum hardware voltage to 18.0V
        self.max_voltage = 5.0  # Default max voltage for RL control (replaces scaling factor)

        # Performance optimization settings - MODIFIED: Optimized UI updates
        self.ui_update_interval = 15  # Increased UI update interval (higher = less UI overhead)
        self.ui_counter = 0
        self.last_loop_time = time.time()
        self.actual_frequency = 0

        # Cache for UI values to avoid unnecessary updates
        self.ui_cache = {
            'motor_angle': 0.0,
            'pendulum_angle': 0.0,
            'rpm': 0.0,
            'voltage': 0.0,
            'frequency': 0.0
        }

        # Performance tracking
        self.iteration_count = 0
        self.start_time = time.time()

        # Data logging variables
        self.logging = False
        self.log_file = None
        self.log_writer = None
        self.log_counter = 0
        self.log_interval = 20  # Log every 20 steps
        self.log_step = 0
        self.log_start_time = 0

        # Pendulum velocity tracking for logging
        self.prev_pendulum_angle = 0
        self.prev_motor_pos = 0
        self.prev_logging_time = time.time()

        # Low-pass filters
        self.filter_cutoff = 500.0  # Default cutoff frequency
        self.pendulum_velocity_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
        self.motor_velocity_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
        self.voltage_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
        self.filtered_voltage = 0.0

        # Initialize the RL model (but don't load weights yet)
        self.initialize_rl_model()

        # Training variables - NEW
        self.training_mode = False
        self.sac_agent = None
        self.replay_buffer = None
        self.episode_rewards = []
        self.episode_reward = 0
        self.episode_step = 0
        self.training_thread = None
        self.stop_training = False
        self.use_filters_during_training = True
        self.random_reset = False
        self.previous_state = None
        self.batch_size = 256
        self.update_freq = 1
        self.episodes_completed = 0
        self.training_start_time = 0

        # Create training params
        self.initialize_training_params()

        # Create GUI elements
        self.create_gui()

    def initialize_training_params(self):
        """Initialize parameters for SAC training"""
        self.training_params = {
            'state_dim': 6,  # Observation dimension
            'action_dim': 1,  # Action dimension
            'hidden_dim': 256,  # Neural network hidden layer size
            'buffer_size': 100000,  # Replay buffer capacity
            'batch_size': 256,  # Training batch size
            'lr': 3e-4,  # Learning rate
            'gamma': 0.99,  # Discount factor
            'tau': 0.005,  # Target network update rate
            'alpha': 0.2,  # Temperature parameter (or learned)
            'auto_entropy': True,  # Automatic entropy tuning
            'updates_per_step': 1,  # Updates per environment step
            'max_episode_steps': 2000  # Maximum steps per episode
        }

    def initialize_rl_model(self):
        """Initialize the RL model architecture"""
        state_dim = 6  # Our observation space (same as in training)
        action_dim = 1  # Motor voltage (normalized)

        # Initialize model
        self.actor = Actor(state_dim, action_dim)

        # Set device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor.eval()  # Set to evaluation mode

    def load_rl_model(self):
        """Open file dialog to select the model file"""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select RL Model File",
            filetypes=(("PyTorch Models", "*.pth"), ("All files", "*.*"))
        )

        if filename:
            try:
                # Load the model state dict
                state_dict = torch.load(filename, map_location=self.device)

                # Try loading the model
                try:
                    self.actor.load_state_dict(state_dict)
                    self.actor.to(self.device)
                    self.actor.eval()
                except Exception as e:
                    raise ValueError(f"Failed to load model: {str(e)}")

                # Confirm model loaded
                self.status_label.config(text=f"Model loaded: {os.path.basename(filename)}")
                self.rl_model = filename

                # Update model loaded indication
                self.model_type_label.config(text="Model Status: Loaded")
                self.architecture_label.config(text="Model ready for control")

                # Enable RL control button
                self.rl_control_btn.config(state=tk.NORMAL)

                # Set blue LED to indicate ready
                self.r_slider.set(0)
                self.g_slider.set(0)
                self.b_slider.set(999)

            except Exception as e:
                self.status_label.config(text=f"Error loading model: {str(e)}")
                messagebox.showerror("Load Error", f"Could not load model: {str(e)}")

    def load_critic_model(self):
        """Open file dialog to select the critic model file"""
        if not hasattr(self, 'sac_agent') or self.sac_agent is None:
            messagebox.showerror("Error", "Initialize SAC agent first by clicking 'Setup Training'")
            return

        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Critic Model File",
            filetypes=(("PyTorch Models", "*.pth"), ("All files", "*.*"))
        )

        if filename:
            try:
                # Load the critic state dict
                state_dict = torch.load(filename, map_location=self.device)

                # Try loading the model
                self.sac_agent.critic.load_state_dict(state_dict)
                self.sac_agent.critic_target.load_state_dict(state_dict)

                # Confirm model loaded
                self.status_label.config(text=f"Critic loaded: {os.path.basename(filename)}")

            except Exception as e:
                self.status_label.config(text=f"Error loading critic: {str(e)}")
                messagebox.showerror("Load Error", f"Could not load critic model: {str(e)}")

    def setup_training(self):
        """Initialize the SAC agent and replay buffer for training"""
        try:
            # Get parameters from UI
            try:
                hidden_dim = int(self.hidden_dim_entry.get())
                buffer_size = int(self.buffer_size_entry.get())
                self.batch_size = int(self.batch_size_entry.get())
                lr = float(self.lr_entry.get())
                gamma = float(self.gamma_entry.get())
                tau = float(self.tau_entry.get())
                alpha = float(self.alpha_entry.get())
                self.update_freq = int(self.update_freq_entry.get())
                max_episode_steps = int(self.max_steps_entry.get())
            except ValueError as e:
                messagebox.showerror("Parameter Error", f"Invalid parameter value: {str(e)}")
                return

            # Update training params
            self.training_params['hidden_dim'] = hidden_dim
            self.training_params['buffer_size'] = buffer_size
            self.training_params['batch_size'] = self.batch_size
            self.training_params['lr'] = lr
            self.training_params['gamma'] = gamma
            self.training_params['tau'] = tau
            self.training_params['alpha'] = alpha
            self.training_params['updates_per_step'] = self.update_freq
            self.training_params['max_episode_steps'] = max_episode_steps
            self.training_params['auto_entropy'] = bool(self.auto_entropy_var.get())

            # Initialize replay buffer
            self.replay_buffer = ReplayBuffer(buffer_size)

            # Initialize SAC agent
            self.sac_agent = SACAgent(
                state_dim=self.training_params['state_dim'],
                action_dim=self.training_params['action_dim'],
                hidden_dim=hidden_dim,
                lr=lr,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                automatic_entropy_tuning=bool(self.auto_entropy_var.get())
            )

            # If we have a loaded actor model, copy its weights to the SAC agent
            if self.rl_model:
                try:
                    self.sac_agent.actor.load_state_dict(self.actor.state_dict())
                    print("Transferred weights from loaded actor to SAC agent")
                except Exception as e:
                    print(f"Warning: Could not transfer weights: {str(e)}")

            # Reset training statistics
            self.episode_rewards = []
            self.episodes_completed = 0

            # Update UI
            self.status_label.config(text="Training setup complete. Ready to start training.")
            self.setup_train_btn.config(text="Reset Training")
            self.train_btn.config(state=tk.NORMAL)
            self.load_critic_btn.config(state=tk.NORMAL)
            self.save_model_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Setup Error", f"Failed to setup training: {str(e)}")
            raise e

    def toggle_training(self):
        """Start or stop the training process"""
        if not self.training_mode:
            # Check if SAC agent is initialized
            if self.sac_agent is None:
                messagebox.showerror("Error", "Please setup training first")
                return

            # Start training
            self.training_mode = True
            self.stop_training = False
            self.train_btn.config(text="Stop Training")

            # Disable other controls
            self.rl_control_btn.config(state=tk.DISABLED)
            self.move_btn.config(state=tk.DISABLED)
            self.calibrate_btn.config(state=tk.DISABLED)
            self.setup_train_btn.config(state=tk.DISABLED)
            self.load_model_btn.config(state=tk.DISABLED)
            self.load_critic_btn.config(state=tk.DISABLED)

            # Set magenta LED during training
            self.r_slider.set(999)
            self.g_slider.set(0)
            self.b_slider.set(999)

            # Reset episode variables
            self.episode_reward = 0
            self.episode_step = 0
            self.previous_state = None
            self.training_start_time = time.time()

            # Get filter setting
            self.use_filters_during_training = bool(self.use_filters_var.get())
            self.random_reset = bool(self.random_reset_var.get())

            # Update status
            self.status_label.config(text="Training active. Calibrating...")

            # Start with calibration
            self.calibrate_for_training()

        else:
            # Stop training
            self.training_mode = False
            self.stop_training = True
            self.train_btn.config(text="Start Training")

            # Re-enable controls
            self.rl_control_btn.config(state=tk.NORMAL)
            self.move_btn.config(state=tk.NORMAL)
            self.calibrate_btn.config(state=tk.NORMAL)
            self.setup_train_btn.config(state=tk.NORMAL)
            self.load_model_btn.config(state=tk.NORMAL)
            self.load_critic_btn.config(state=tk.NORMAL)

            # Stop motor and set blue LED
            self.motor_voltage = 0.0
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

            # Update status
            self.status_label.config(text="Training stopped")

    def calibrate_for_training(self):
        """Calibrate before starting training episode"""
        # Set yellow LED during calibration
        self.r_slider.set(999)
        self.g_slider.set(999)
        self.b_slider.set(0)

        # Apply voltage to move to corner
        self.motor_voltage = 1.5
        self.calibrating = True
        self.calibration_start_time = time.time()

    def update_training(self):
        """Main training loop - called on each iteration"""
        # First, check if we're calibrating
        if self.calibrating:
            elapsed = time.time() - self.calibration_start_time

            if elapsed < 3.0:
                # Still calibrating
                if self.ui_counter == 0:
                    self.status_label.config(text=f"Calibrating... ({3.0 - elapsed:.1f}s)")
            else:
                # Calibration complete
                self.motor_voltage = 0.0
                self.qube.resetMotorEncoder()
                self.qube.resetPendulumEncoder()
                self.calibrating = False

                # Set magenta LED for training
                self.r_slider.set(999)
                self.g_slider.set(0)
                self.b_slider.set(999)

                # Initialize pendulum in random or default position
                if self.random_reset:
                    # Random initial position within safe range
                    random_voltage = np.random.uniform(-1.0, 1.0)
                    self.motor_voltage = random_voltage

                    # Apply for a short duration
                    time.sleep(0.3)
                    self.motor_voltage = 0.0

                # Reset episode variables
                self.episode_reward = 0
                self.episode_step = 0

                # Get initial state
                self.previous_state = self.get_observation()[0]

                # Inform user
                self.status_label.config(text=f"Training episode {self.episodes_completed + 1} started")
                return

        # If we have a previous state, we're in the middle of an episode
        if self.previous_state is not None:
            # Check if episode done
            if self.episode_step >= self.training_params['max_episode_steps']:
                # Episode complete
                self.episodes_completed += 1
                self.episode_rewards.append(self.episode_reward)

                # Calculate average reward
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(
                    self.episode_rewards)

                # Update status
                self.status_label.config(
                    text=f"Episode {self.episodes_completed} complete. Reward: {self.episode_reward:.2f}, Avg: {avg_reward:.2f}")

                # Log to console
                elapsed = time.time() - self.training_start_time
                print(
                    f"Episode {self.episodes_completed} - Steps: {self.episode_step}, Reward: {self.episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Time: {elapsed:.1f}s")

                # Update reward display
                self.reward_label.config(text=f"{self.episode_reward:.2f}")
                self.avg_reward_label.config(text=f"{avg_reward:.2f}")
                self.episodes_label.config(text=f"{self.episodes_completed}")

                # Reset for next episode
                self.previous_state = None
                self.motor_voltage = 0.0

                # Start calibration for next episode
                self.calibrate_for_training()
                return

            # Select action from policy (with exploration)
            action = self.sac_agent.select_action(self.previous_state, evaluate=False)

            # Scale action to voltage
            self.motor_voltage = float(action[0]) * self.max_voltage

            # Apply action and get new state
            current_obs, pendulum_angle_norm = self.get_observation()

            # Compute reward
            reward = self._compute_reward(pendulum_angle_norm)

            # Update episode reward
            self.episode_reward += reward

            # Periodically update UI with current reward
            if self.ui_counter == 0:
                self.reward_label.config(text=f"{self.episode_reward:.2f}")

                # Classify state for user feedback
                upright_angle = abs(pendulum_angle_norm) * 180 / np.pi
                if upright_angle < 30:
                    self.status_label.config(
                        text=f"Training: Ep {self.episodes_completed + 1}, Step {self.episode_step}, Near balance ({upright_angle:.1f}°)")
                else:
                    self.status_label.config(
                        text=f"Training: Ep {self.episodes_completed + 1}, Step {self.episode_step}, Swinging ({upright_angle:.1f}°)")

            # Check terminal condition (pendulum fell or arm at limit)
            # In real system, we don't terminate episodes early to maximize learning
            done = False

            # Store transition in replay buffer
            self.replay_buffer.push(
                self.previous_state,
                action,
                reward,
                current_obs,
                done
            )

            # Update for next step
            self.previous_state = current_obs
            self.episode_step += 1

            # Update networks if enough samples & correct update interval
            if len(self.replay_buffer) > self.batch_size and self.episode_step % self.update_freq == 0:
                for _ in range(self.training_params['updates_per_step']):
                    self.sac_agent.update_parameters(self.replay_buffer, self.batch_size)

    def _compute_reward(self, pendulum_angle_norm):
        """Calculate reward based on current state"""
        # Get state values
        theta_0, theta_1, theta_0_dot, theta_1_dot = self.state

        # We'll use a simplified version of the reward function from SimRL

        # 1. Base reward for pendulum being upright (range: -1 to 1)
        upright_reward = 1.0 * np.cos(pendulum_angle_norm)

        # 2. Penalty for arm position away from center
        pos_penalty = -abs(theta_0) / 2.0

        # 3. Bonus for being close to upright position
        arm_center = np.exp(-1.0 * theta_0 ** 2)
        upright_closeness = np.exp(-10.0 * pendulum_angle_norm ** 2)
        stability_factor = np.exp(-0.6 * theta_1_dot ** 2)
        bonus = 3.0 * upright_closeness * stability_factor * arm_center

        # 4. Energy management reward
        energy_reward = 2 - 0.007 * (theta_1_dot ** 2)

        # Combine all components
        reward = (
                upright_reward +
                pos_penalty +
                bonus +
                energy_reward
        )

        return reward

    def save_trained_model(self):
        """Save the trained actor and critic models"""
        if not hasattr(self, 'sac_agent') or self.sac_agent is None:
            messagebox.showerror("Error", "No trained model to save")
            return

        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save actor model
            actor_filename = f"trained_actor_{timestamp}.pth"
            torch.save(self.sac_agent.actor.state_dict(), actor_filename)

            # Save critic model
            critic_filename = f"trained_critic_{timestamp}.pth"
            torch.save(self.sac_agent.critic.state_dict(), critic_filename)

            # Also update the main actor for immediate use
            self.actor.load_state_dict(self.sac_agent.actor.state_dict())
            self.actor.eval()

            # Update UI
            self.status_label.config(text=f"Models saved: {actor_filename} and {critic_filename}")
            self.rl_model = actor_filename  # Update the main model reference

            # Enable RL control with the new model
            self.rl_control_btn.config(state=tk.NORMAL)

            messagebox.showinfo("Save Success",
                                f"Models saved successfully.\nActor: {actor_filename}\nCritic: {critic_filename}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save models: {str(e)}")

    def create_gui(self):
        # Main control frame
        control_frame = Frame(self.master, padx=10, pady=10)
        control_frame.pack()

        # Calibrate button
        self.calibrate_btn = Button(control_frame, text="Calibrate (Set Corner as Zero)",
                                    command=self.calibrate,
                                    width=25, height=2)
        self.calibrate_btn.grid(row=0, column=0, padx=5, pady=5)

        # RL Model buttons
        rl_frame = Frame(control_frame)
        rl_frame.grid(row=1, column=0, pady=10)

        self.load_model_btn = Button(rl_frame, text="Load RL Model",
                                     command=self.load_rl_model,
                                     width=15)
        self.load_model_btn.grid(row=0, column=0, padx=5)

        self.rl_control_btn = Button(rl_frame, text="Start RL Control",
                                     command=self.toggle_rl_control,
                                     width=15, state=tk.DISABLED)
        self.rl_control_btn.grid(row=0, column=1, padx=5)

        # Model Status Indicator
        self.model_type_label = Label(rl_frame, text="Model Status: Not Loaded", width=20)
        self.model_type_label.grid(row=0, column=2, padx=5)

        # Max voltage slider
        max_voltage_frame = Frame(control_frame)
        max_voltage_frame.grid(row=2, column=0, pady=5)

        self.max_voltage_slider = Scale(
            max_voltage_frame,
            from_=0.5,
            to=self.system_max_voltage,
            orient=tk.HORIZONTAL,
            label="RL Max Voltage",
            length=300,
            resolution=0.1,
            command=self.set_max_voltage
        )
        self.max_voltage_slider.set(self.max_voltage)
        self.max_voltage_slider.pack(padx=5)

        # Move to position input and button
        position_frame = Frame(control_frame)
        position_frame.grid(row=3, column=0, pady=10)

        Label(position_frame, text="Target Position (degrees):").grid(row=0, column=0, padx=5)
        self.position_entry = Entry(position_frame, width=10)
        self.position_entry.grid(row=0, column=1, padx=5)
        self.position_entry.insert(0, "0.0")

        self.move_btn = Button(position_frame, text="Move to Position",
                               command=self.start_move_to_position, width=15)
        self.move_btn.grid(row=0, column=2, padx=5)

        # TRAINING SECTION - NEW
        # Create a LabelFrame for training settings
        train_frame = Frame(self.master, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        train_frame.pack(pady=10, fill=tk.X, padx=10)

        # Title label
        Label(train_frame, text="Training Settings", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=4,
                                                                                      pady=5)

        # Left column - Parameters
        param_frame = Frame(train_frame)
        param_frame.grid(row=1, column=0, padx=10, sticky="n")

        # SAC Parameters
        Label(param_frame, text="Neural Network Size:").grid(row=0, column=0, sticky="w")
        self.hidden_dim_entry = Entry(param_frame, width=8)
        self.hidden_dim_entry.grid(row=0, column=1, pady=2)
        self.hidden_dim_entry.insert(0, "256")

        Label(param_frame, text="Buffer Size:").grid(row=1, column=0, sticky="w")
        self.buffer_size_entry = Entry(param_frame, width=8)
        self.buffer_size_entry.grid(row=1, column=1, pady=2)
        self.buffer_size_entry.insert(0, "100000")

        Label(param_frame, text="Batch Size:").grid(row=2, column=0, sticky="w")
        self.batch_size_entry = Entry(param_frame, width=8)
        self.batch_size_entry.grid(row=2, column=1, pady=2)
        self.batch_size_entry.insert(0, "256")

        Label(param_frame, text="Learning Rate:").grid(row=3, column=0, sticky="w")
        self.lr_entry = Entry(param_frame, width=8)
        self.lr_entry.grid(row=3, column=1, pady=2)
        self.lr_entry.insert(0, "0.0003")

        # Middle column - More parameters
        param_frame2 = Frame(train_frame)
        param_frame2.grid(row=1, column=1, padx=10, sticky="n")

        Label(param_frame2, text="Discount (gamma):").grid(row=0, column=0, sticky="w")
        self.gamma_entry = Entry(param_frame2, width=8)
        self.gamma_entry.grid(row=0, column=1, pady=2)
        self.gamma_entry.insert(0, "0.99")

        Label(param_frame2, text="Tau:").grid(row=1, column=0, sticky="w")
        self.tau_entry = Entry(param_frame2, width=8)
        self.tau_entry.grid(row=1, column=1, pady=2)
        self.tau_entry.insert(0, "0.005")

        Label(param_frame2, text="Alpha:").grid(row=2, column=0, sticky="w")
        self.alpha_entry = Entry(param_frame2, width=8)
        self.alpha_entry.grid(row=2, column=1, pady=2)
        self.alpha_entry.insert(0, "0.2")

        Label(param_frame2, text="Updates/Step:").grid(row=3, column=0, sticky="w")
        self.update_freq_entry = Entry(param_frame2, width=8)
        self.update_freq_entry.grid(row=3, column=1, pady=2)
        self.update_freq_entry.insert(0, "1")

        # Right column - Additional settings
        settings_frame = Frame(train_frame)
        settings_frame.grid(row=1, column=2, padx=10, sticky="n")

        Label(settings_frame, text="Max Episode Steps:").grid(row=0, column=0, sticky="w")
        self.max_steps_entry = Entry(settings_frame, width=8)
        self.max_steps_entry.grid(row=0, column=1, pady=2)
        self.max_steps_entry.insert(0, "2000")

        # Checkboxes
        self.auto_entropy_var = IntVar(value=1)
        self.use_filters_var = IntVar(value=1)
        self.random_reset_var = IntVar(value=0)

        Checkbutton(settings_frame, text="Auto Entropy", variable=self.auto_entropy_var).grid(row=1, column=0,
                                                                                              columnspan=2, sticky="w",
                                                                                              pady=2)
        Checkbutton(settings_frame, text="Use Filters During Training", variable=self.use_filters_var).grid(row=2,
                                                                                                            column=0,
                                                                                                            columnspan=2,
                                                                                                            sticky="w",
                                                                                                            pady=2)
        Checkbutton(settings_frame, text="Random Initial Position", variable=self.random_reset_var).grid(row=3,
                                                                                                         column=0,
                                                                                                         columnspan=2,
                                                                                                         sticky="w",
                                                                                                         pady=2)

        # Training control buttons
        button_frame = Frame(train_frame)
        button_frame.grid(row=1, column=3, padx=10, sticky="n")

        self.setup_train_btn = Button(button_frame, text="Setup Training",
                                      command=self.setup_training, width=15)
        self.setup_train_btn.grid(row=0, column=0, pady=2)

        self.train_btn = Button(button_frame, text="Start Training",
                                command=self.toggle_training, width=15, state=tk.DISABLED)
        self.train_btn.grid(row=1, column=0, pady=2)

        self.load_critic_btn = Button(button_frame, text="Load Critic",
                                      command=self.load_critic_model, width=15, state=tk.DISABLED)
        self.load_critic_btn.grid(row=2, column=0, pady=2)

        self.save_model_btn = Button(button_frame, text="Save Models",
                                     command=self.save_trained_model, width=15, state=tk.DISABLED)
        self.save_model_btn.grid(row=3, column=0, pady=2)

        # Training stats display
        stats_frame = Frame(train_frame)
        stats_frame.grid(row=2, column=0, columnspan=4, pady=10)

        Label(stats_frame, text="Training Stats:", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=6,
                                                                                    pady=5)

        Label(stats_frame, text="Episodes:").grid(row=1, column=0, padx=5)
        self.episodes_label = Label(stats_frame, text="0", width=8)
        self.episodes_label.grid(row=1, column=1, padx=5)

        Label(stats_frame, text="Current Reward:").grid(row=1, column=2, padx=5)
        self.reward_label = Label(stats_frame, text="0.0", width=8)
        self.reward_label.grid(row=1, column=3, padx=5)

        Label(stats_frame, text="Avg Reward:").grid(row=1, column=4, padx=5)
        self.avg_reward_label = Label(stats_frame, text="0.0", width=8)
        self.avg_reward_label.grid(row=1, column=5, padx=5)

        # FILTER SECTION
        filter_frame = Frame(control_frame)
        filter_frame.grid(row=5, column=0, pady=5)

        # New filter cutoff slider (replacing control frequency slider)
        self.filter_cutoff_slider = Scale(
            filter_frame,
            from_=0,
            to=6000,
            orient=tk.HORIZONTAL,
            label="Filter Cutoff Frequency (Hz)",
            length=300,
            resolution=100,
            command=self.set_filter_cutoff
        )
        self.filter_cutoff_slider.set(self.filter_cutoff)
        self.filter_cutoff_slider.grid(row=0, column=0, padx=5)

        # Filter status
        self.filter_status_label = Label(filter_frame, text=f"Cutoff: {self.filter_cutoff} Hz")
        self.filter_status_label.grid(row=0, column=1, padx=5)

        # Stop button
        self.stop_btn = Button(control_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=20, height=2,
                               bg="red", fg="white")
        self.stop_btn.grid(row=6, column=0, pady=10)

        # Manual voltage control
        self.voltage_slider = Scale(
            control_frame,
            from_=-self.system_max_voltage,
            to=self.system_max_voltage,
            orient=tk.HORIZONTAL,
            label="Manual Voltage",
            length=400,
            resolution=0.1,
            command=self.set_manual_voltage
        )
        self.voltage_slider.set(0)
        self.voltage_slider.grid(row=7, column=0, padx=5, pady=10)

        # Performance settings frame - MODIFIED: Only UI update interval
        perf_frame = Frame(control_frame)
        perf_frame.grid(row=8, column=0, pady=5)

        # Add UI update interval slider with increased range
        self.ui_slider = Scale(
            perf_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            label="UI Update Every N Iterations",
            length=300,
            resolution=5,
            command=self.set_ui_update_interval
        )
        self.ui_slider.set(self.ui_update_interval)
        self.ui_slider.grid(row=0, column=0, padx=5)

        # Status display
        status_frame = Frame(self.master, padx=10, pady=10)
        status_frame.pack()

        Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = Label(status_frame, text="Ready - Please calibrate", width=40)
        self.status_label.grid(row=0, column=1, sticky=tk.W)

        Label(status_frame, text="Model:").grid(row=1, column=0, sticky=tk.W)
        self.model_label = Label(status_frame, text="No model loaded", width=40)
        self.model_label.grid(row=1, column=1, sticky=tk.W)

        Label(status_frame, text="Architecture:").grid(row=2, column=0, sticky=tk.W)
        self.architecture_label = Label(status_frame, text="Not loaded", width=40)
        self.architecture_label.grid(row=2, column=1, sticky=tk.W)

        Label(status_frame, text="Motor Angle:").grid(row=3, column=0, sticky=tk.W)
        self.angle_label = Label(status_frame, text="0.0°")
        self.angle_label.grid(row=3, column=1, sticky=tk.W)

        Label(status_frame, text="Pendulum Angle:").grid(row=4, column=0, sticky=tk.W)
        self.pendulum_label = Label(status_frame, text="0.0°")
        self.pendulum_label.grid(row=4, column=1, sticky=tk.W)

        Label(status_frame, text="Motor RPM:").grid(row=5, column=0, sticky=tk.W)
        self.rpm_label = Label(status_frame, text="0.0")
        self.rpm_label.grid(row=5, column=1, sticky=tk.W)

        Label(status_frame, text="Current Voltage:").grid(row=6, column=0, sticky=tk.W)
        self.voltage_label = Label(status_frame, text="0.0 V")
        self.voltage_label.grid(row=6, column=1, sticky=tk.W)

        Label(status_frame, text="RL Max Voltage:").grid(row=7, column=0, sticky=tk.W)
        self.max_voltage_label = Label(status_frame, text=f"{self.max_voltage:.1f} V")
        self.max_voltage_label.grid(row=7, column=1, sticky=tk.W)

        # Add actual frequency display
        Label(status_frame, text="Control Frequency:").grid(row=8, column=0, sticky=tk.W)
        self.freq_label = Label(status_frame, text="0.0 Hz")
        self.freq_label.grid(row=8, column=1, sticky=tk.W)

        # RGB Control
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack()

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

    # MODIFIED: New method for filter cutoff slider
    def set_filter_cutoff(self, value):
        """Set the cutoff frequency of all filters from slider"""
        try:
            cutoff_freq = float(value)
            if cutoff_freq <= 0:
                cutoff_freq = 100.0  # Minimum value for safety

            # Store current filter values to preserve states
            pendulum_velocity_last = self.pendulum_velocity_filter.y_last
            motor_velocity_last = self.motor_velocity_filter.y_last
            voltage_last = self.voltage_filter.y_last

            # Update internal value
            self.filter_cutoff = cutoff_freq

            # Create new filters with updated cutoff frequency
            self.pendulum_velocity_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
            self.motor_velocity_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
            self.voltage_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)

            # Transfer the previous states to maintain continuity
            self.pendulum_velocity_filter.reset(pendulum_velocity_last)
            self.motor_velocity_filter.reset(motor_velocity_last)
            self.voltage_filter.reset(voltage_last)

            # Update status label
            self.filter_status_label.config(text=f"Cutoff: {self.filter_cutoff} Hz")

        except ValueError:
            # Handle invalid input
            self.filter_status_label.config(text="Invalid cutoff value!")

    def set_ui_update_interval(self, value):
        """Set UI update interval from slider"""
        self.ui_update_interval = int(value)

    def set_max_voltage(self, value):
        """Set the maximum voltage for RL control from slider"""
        self.max_voltage = float(value)
        self.max_voltage_label.config(text=f"{self.max_voltage:.1f} V")

    def set_manual_voltage(self, value):
        """Set manual voltage from slider"""
        self.motor_voltage = float(value)
        # Reset any automatic control modes
        self.calibrating = False
        self.moving_to_position = False
        self.rl_mode = False
        self.training_mode = False

        if self.rl_model:
            self.rl_control_btn.config(text="Start RL Control")

    def calibrate(self):
        """Simple calibration - move to corner and set as zero"""
        self.calibrating = True
        self.moving_to_position = False
        self.rl_mode = False
        self.training_mode = False
        self.voltage_slider.set(0)  # Reset slider

        # Set calibration start time
        self.calibration_start_time = time.time()
        self.status_label.config(text="Calibrating - Finding corner...")

        # Set yellow LED during calibration
        self.r_slider.set(999)
        self.g_slider.set(999)
        self.b_slider.set(0)

    def update_calibration(self):
        """Update calibration process"""
        elapsed = time.time() - self.calibration_start_time

        if elapsed < 3.0:
            # Apply voltage to move to corner
            self.motor_voltage = 1.5
            if self.ui_counter == 0:  # Only update UI when counter is 0
                self.status_label.config(text=f"Finding corner... ({3.0 - elapsed:.1f}s)")
        else:
            # At corner - set as zero
            self.motor_voltage = 0.0

            # Reset encoders at the corner position
            self.qube.resetMotorEncoder()
            self.qube.resetPendulumEncoder()

            # Set blue LED to indicate ready
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

            # End calibration
            self.calibrating = False
            self.status_label.config(text="Calibration complete - Corner is now zero")

    def toggle_rl_control(self):
        """Toggle RL control mode on/off"""
        if not self.rl_mode:
            # Start RL control
            self.rl_mode = True
            self.moving_to_position = False
            self.calibrating = False
            self.training_mode = False
            self.voltage_slider.set(0)  # Reset slider
            self.rl_control_btn.config(text="Stop RL Control")
            self.status_label.config(text="RL control active")

            # Set purple LED during RL control
            self.r_slider.set(500)
            self.g_slider.set(0)
            self.b_slider.set(999)

        else:
            # Stop RL control
            self.rl_mode = False
            self.motor_voltage = 0.0
            self.rl_control_btn.config(text="Start RL Control")
            self.status_label.config(text="RL control stopped")

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def start_move_to_position(self):
        """Start moving to target position"""
        if not self.calibrating and not self.rl_mode and not self.training_mode:
            try:
                # Get target position from entry field
                self.target_position = float(self.position_entry.get())

                self.moving_to_position = True
                self.voltage_slider.set(0)  # Reset slider
                self.status_label.config(text=f"Moving to {self.target_position:.1f}°...")

                # Set green LED during movement
                self.r_slider.set(0)
                self.g_slider.set(999)
                self.b_slider.set(0)
            except ValueError:
                self.status_label.config(text="Invalid position value")

    def update_position_control(self):
        """Update position control"""
        # Get current angle
        current_angle = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero

        # Calculate position error (difference from target)
        position_error = self.target_position - current_angle

        # Check if we're close enough
        if abs(position_error) < 0.5:
            # We've reached the target
            self.moving_to_position = False
            self.motor_voltage = 0.0

            # Set blue LED when done
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

            if self.ui_counter == 0:  # Only update UI when counter is 0
                self.status_label.config(text="Position reached")
            return

        # Simple proportional control
        kp = 0.02  # Low gain
        self.motor_voltage = kp * position_error

        # Limit voltage for safety
        self.motor_voltage = max(-self.system_max_voltage, min(self.system_max_voltage, self.motor_voltage))

    def get_observation(self):
        """Get the current observation vector"""
        # Get current state
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert all angles to radians for the RL model
        motor_angle = np.radians(motor_angle_deg)

        # For pendulum angle, convert to radians and flip convention:
        # QUBE: 0 degrees = down position
        # Training: 0 radians = upright position, π radians = down position
        pendulum_angle = np.radians(pendulum_angle_deg)
        # Adjust so that upright is 0
        pendulum_angle_norm = normalize_angle(pendulum_angle + np.pi)

        # Estimate motor velocity - convert from RPM to rad/s
        motor_velocity = self.qube.getMotorRPM() * (2 * np.pi / 60)  # Convert RPM to rad/s

        # Estimate pendulum velocity by finite difference (in radians)
        current_pendulum_angle_rad = np.radians(pendulum_angle_deg)
        current_time = time.time()

        if not hasattr(self, 'prev_pendulum_angle_rl'):
            self.prev_pendulum_angle_rl = current_pendulum_angle_rad
            self.prev_time_rl = current_time
            pendulum_velocity = 0.0
        else:
            dt = current_time - self.prev_time_rl
            if dt > 0:
                raw_velocity = (current_pendulum_angle_rad - self.prev_pendulum_angle_rl) / dt

                # Apply low-pass filter to pendulum velocity for RL
                # Only if we're using filters during training
                if self.use_filters_during_training or not self.training_mode:
                    pendulum_velocity = self.pendulum_velocity_filter.filter(raw_velocity, dt)
                else:
                    pendulum_velocity = raw_velocity
            else:
                pendulum_velocity = 0.0
            self.prev_pendulum_angle_rl = current_pendulum_angle_rad
            self.prev_time_rl = current_time

        # Save state for reward calculation
        self.state = np.array([
            motor_angle, pendulum_angle_norm, motor_velocity, pendulum_velocity
        ])

        # Create observation vector
        obs = np.array([
            np.sin(motor_angle), np.cos(motor_angle),
            np.sin(pendulum_angle_norm), np.cos(pendulum_angle_norm),
            motor_velocity / 10.0,  # Scale velocities as in training
            pendulum_velocity / 10.0
        ])

        # Return both the observation and pendulum angle for status updates
        return obs, pendulum_angle_norm

    def update_rl_control(self):
        """Update RL control logic"""
        # Get current state
        obs, pendulum_angle_norm = self.get_observation()

        # Get action from RL model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            action_mean, _ = self.actor(state_tensor)
            action = action_mean.cpu().numpy()[0][0]  # Get action as scalar

        # Apply max_voltage directly
        raw_voltage = float(action) * self.max_voltage

        # Apply low-pass filter to voltage
        current_time = time.time()
        dt = current_time - self.last_loop_time if hasattr(self, 'last_loop_time') else 0.001
        self.motor_voltage = self.voltage_filter.filter(raw_voltage, dt)

        # Update status - but only during UI updates
        if self.ui_counter == 0:
            upright_angle = abs(pendulum_angle_norm) * 180 / np.pi
            if upright_angle < 30:
                self.status_label.config(text=f"Control active - Near balance ({upright_angle:.1f}° from upright)")
            else:
                self.status_label.config(text=f"Control active - Swinging ({upright_angle:.1f}° from upright)")

    def stop_motor(self):
        """Stop the motor"""
        self.calibrating = False
        self.moving_to_position = False
        self.rl_mode = False
        self.training_mode = False
        self.motor_voltage = 0.0
        self.voltage_slider.set(0)

        if self.rl_model:
            self.rl_control_btn.config(text="Start RL Control")

        if hasattr(self, 'train_btn'):
            self.train_btn.config(text="Start Training")

        # Set blue LED when stopped
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(999)

        self.status_label.config(text="Motor stopped")

    def update_gui(self):
        """Update the GUI and control the hardware - called continuously"""
        # Increment iteration counter
        self.iteration_count += 1

        # Calculate current control frequency (less frequently)
        now = time.time()
        elapsed = now - self.last_loop_time
        self.last_loop_time = now

        # Smooth frequency calculation (moving average)
        alpha = 0.05  # Reduced smoothing factor for less processing
        self.actual_frequency = (1 - alpha) * self.actual_frequency + alpha * (1.0 / max(elapsed, 0.001))

        # Update UI counter
        self.ui_counter = (self.ui_counter + 1) % self.ui_update_interval

        # CRITICAL CONTROL OPERATIONS - These must happen on every loop
        # ---------------------------------------------------------------

        # Update automatic control modes if active (core functionality)
        if self.calibrating:
            self.update_calibration()
        elif self.moving_to_position:
            self.update_position_control()
        elif self.rl_mode:
            self.update_rl_control()
        elif self.training_mode:
            self.update_training()

        # Apply the current motor voltage - CRITICAL TO DO ON EVERY LOOP!
        self.qube.setMotorVoltage(self.motor_voltage)

        # Only update RGB values periodically to reduce hardware communication overhead
        if self.ui_counter % 3 == 0:  # Update RGB less frequently than UI
            self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

        # NON-CRITICAL UI UPDATES - Only do these periodically
        # ---------------------------------------------------------------
        if self.ui_counter == 0:
            # Get current hardware values
            motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero
            pendulum_angle_deg = self.qube.getPendulumAngle()
            pendulum_angle_rad = np.radians(pendulum_angle_deg)
            pendulum_angle_norm = normalize_angle(pendulum_angle_rad + np.pi)
            rpm = self.qube.getMotorRPM()

            # Check if values have changed enough to warrant a UI update
            # This avoids unnecessary tkinter operations which are expensive
            if (abs(motor_angle_deg - self.ui_cache['motor_angle']) > 0.5 or
                    abs(pendulum_angle_deg - self.ui_cache['pendulum_angle']) > 0.5):
                # Update angle displays
                self.angle_label.config(text=f"{motor_angle_deg:.1f}°")
                self.pendulum_label.config(
                    text=f"{pendulum_angle_deg:.1f}° ({abs(pendulum_angle_norm) * 180 / np.pi:.1f}° from upright)")
                # Update cache
                self.ui_cache['motor_angle'] = motor_angle_deg
                self.ui_cache['pendulum_angle'] = pendulum_angle_deg

            # Update other displays only if they've changed significantly
            if abs(rpm - self.ui_cache['rpm']) > 1.0:
                self.rpm_label.config(text=f"{rpm:.1f}")
                self.ui_cache['rpm'] = rpm

            if abs(self.motor_voltage - self.ui_cache['voltage']) > 0.1:
                self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")
                self.ui_cache['voltage'] = self.motor_voltage

            # Update frequency display less often (every 3rd UI update)
            if self.iteration_count % (self.ui_update_interval * 3) == 0:
                self.freq_label.config(text=f"{self.actual_frequency:.1f} Hz")
                self.ui_cache['frequency'] = self.actual_frequency

            # Log performance stats every 1000 iterations instead of 500
            if self.iteration_count % 1000 == 0:
                avg_freq = self.iteration_count / (now - self.start_time)
                print(
                    f"Performance: {self.iteration_count} iterations, Avg frequency: {avg_freq:.1f} Hz, Current: {self.actual_frequency:.1f} Hz")


def main():
    print("Starting QUBE Controller with RL Training and Real-Time Control")
    print("Will set corner position as zero")
    print("OPTIMIZED VERSION - Maximum speed control, real-time training, improved filter control")

    try:
        # Initialize QUBE
        qube = QUBE(COM_PORT, 115200)

        # Initial reset
        qube.resetMotorEncoder()
        qube.resetPendulumEncoder()
        qube.setMotorVoltage(0.0)
        qube.setRGB(0, 0, 999)  # Blue LED to start
        qube.update()

        # Configure Tkinter for higher performance
        root = tk.Tk()

        # Optimize Tkinter settings
        root.update_idletasks()  # Process any pending UI events before starting main loop
        root.protocol("WM_DELETE_WINDOW", root.destroy)  # Ensure clean shutdown

        # Create application
        app = QUBEControllerWithRL(root, qube)

        # Set initial values
        app.filter_cutoff_slider.set(500)  # Default filter cutoff to 500 Hz
        app.max_voltage_slider.set(4.0)  # Set default max voltage
        app.set_max_voltage(4.0)  # Update the internal value

        # Preallocate variables used in main loop
        ui_counter = 0
        update_count = 0

        # Display initial performance advice
        print("Performance tip: Increase 'UI Update Every N Iterations' slider for higher control frequency")

        # Main loop - MODIFIED: High performance optimized loop
        while True:
            try:
                # Update hardware - critical control operation
                qube.update()

                # Update controller - critical control operation
                app.update_gui()

                # Update Tkinter less frequently - major performance improvement
                if app.ui_counter == 0:
                    # Process Tkinter events without full redraw
                    root.update_idletasks()  # Process pending events

                    # Full update only every few UI cycles for better performance
                    update_count += 1
                    if update_count % 3 == 0:  # Reduce full UI updates further
                        root.update()  # Complete update including redraw

            except RuntimeError as e:
                # Handle transient Tkinter errors (can happen during heavy load)
                if "main thread is not in main loop" in str(e):
                    print("Handled UI threading issue, continuing...")
                    pass
                else:
                    raise  # Re-raise if it's a different error

    except tk.TclError:
        # Window closed
        print("Window closed, shutting down")
    except KeyboardInterrupt:
        # User pressed Ctrl+C
        print("Keyboard interrupt, shutting down")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Final stop attempt with more robust shutdown
        try:
            print("Shutting down motor and hardware...")
            qube.setMotorVoltage(0.0)
            qube.setRGB(0, 0, 0)  # Turn off LEDs
            qube.update()
            time.sleep(0.1)  # Brief pause to ensure commands are sent
            print("Motor stopped")
        except Exception as e:
            print(f"Warning: Could not cleanly shut down hardware: {str(e)}")


if __name__ == "__main__":
    main()