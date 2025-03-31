import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog, Checkbutton, IntVar, StringVar
from QUBE import QUBE
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
from datetime import datetime

# Update with your COM port
COM_PORT = "COM10"

# Improved SAC Hyperparameters
BATCH_SIZE = 256  # Increased from 128 for more stable updates
GAMMA = 0.99
TAU = 0.001  # Reduced for more stable target updates
LR = 1e-4  # Reduced from 3e-4 for more careful learning
ALPHA = 0.1  # Reduced from 0.2 for less random exploration initially
AUTO_ENTROPY_TUNING = True
MAX_VOLTAGE = 10.0


# Actor Network - same architecture as in training script
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


# Improved Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # How much prioritization to use

        # Progressive beta for importance sampling
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 1

    def get_beta(self):
        # Calculate beta based on training progress
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        self.frame_idx += 1
        return beta

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []

        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:len(self.buffer)]
        else:
            priorities = self.priorities

        # Calculate sampling probabilities from priorities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -beta
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(
            dones), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


# Helper functions
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


class QUBEControllerWithRLTraining:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        master.title("QUBE Controller with Reinforcement Learning Training")

        # Control state
        self.motor_voltage = 0.0
        self.target_position = 0.0
        self.calibrating = False
        self.rl_mode = False
        self.training_mode = False
        self.moving_to_position = False
        self.rl_model = None
        self.max_voltage = MAX_VOLTAGE
        self.rl_scaling_factor = 6.0
        self.balance_angle_threshold = np.radians(30)  # Default balance threshold

        # Training state variables
        self.experience_counter = 0
        self.update_counter = 0
        self.prev_state = None
        self.prev_action = None
        self.exploration_noise = 0.1  # Exploration noise during training

        # Tracking variables for improved balance detection
        self.near_balance_time = 0.0
        self.last_balance_check = time.time()
        self.was_near_balance = False

        # Variables for dynamic exploration and recovery
        self.exploration_decay = 0.9997  # Decay rate for exploration
        self.min_exploration = 0.05  # Minimum exploration level

        # Variables for curriculum learning
        self.curriculum_stage = 0
        self.curriculum_success_counter = 0
        self.curriculum_failure_counter = 0

        # Variables for checkpointing and recovery
        self.best_episode_reward = -float('inf')
        self.best_episode_length = 0
        self.consecutive_short_episodes = 0

        # RL components
        self.state_dim = 6  # Our observation space
        self.action_dim = 1  # Motor voltage (normalized)
        self.replay_buffer_size = 100000
        self.min_buffer_size_for_training = BATCH_SIZE * 3
        self.training_steps_per_update = 1
        self.update_every_n_steps = 5
        self.init_rl_components()

        # Create GUI elements
        self.create_gui()

        # Stats for training
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode = 0
        self.episode_start_time = time.time()

        # Set up exploration cycle parameters
        self.exploration_cycle_period = 200  # Episodes per exploration cycle
        self.exploration_base = 0.2
        self.exploration_amplitude = 0.2

    def init_rl_components(self):
        """Initialize RL components with better weight initialization"""
        # Initialize SAC components
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)

        # Apply custom weight initialization for more stable start
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)

        self.actor.apply(weights_init)
        self.critic.apply(weights_init)

        # Copy parameters to target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Set device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        # Initialize optimizers with weight decay for regularization
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR, weight_decay=1e-5)

        # Automatic entropy tuning
        if AUTO_ENTROPY_TUNING:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.log_alpha = self.log_alpha.to(self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)
            self.alpha = torch.exp(self.log_alpha).item()
        else:
            self.alpha = ALPHA

        # Set networks to evaluation mode initially
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

        # Initialize prioritized experience replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_size)

    def create_gui(self):
        # Main container frame
        main_frame = Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame (left column)
        control_frame = Frame(main_frame, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Training frame (right column)
        training_frame = Frame(main_frame, padx=10, pady=10, bd=2, relief=tk.RIDGE)
        training_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Status frame (bottom)
        status_frame = Frame(self.master, padx=10, pady=10)
        status_frame.pack(fill=tk.X, expand=False)

        # RGB frame (bottom)
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack(fill=tk.X, expand=False)

        # ---------- Control Frame Elements ----------
        Label(control_frame, text="QUBE Control", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                   pady=5)

        # Calibrate button
        self.calibrate_btn = Button(control_frame, text="Calibrate (Set Corner as Zero)",
                                    command=self.calibrate,
                                    width=25, height=2)
        self.calibrate_btn.grid(row=1, column=0, padx=5, pady=5)

        # RL Model buttons
        rl_frame = Frame(control_frame)
        rl_frame.grid(row=2, column=0, pady=10)

        self.load_model_btn = Button(rl_frame, text="Load RL Model",
                                     command=self.load_rl_model,
                                     width=15)
        self.load_model_btn.grid(row=0, column=0, padx=5)

        self.rl_control_btn = Button(rl_frame, text="Start RL Control",
                                     command=self.toggle_rl_control,
                                     width=15, state=tk.DISABLED)
        self.rl_control_btn.grid(row=0, column=1, padx=5)

        # RL SCALING FACTOR SLIDER
        rl_scaling_frame = Frame(control_frame)
        rl_scaling_frame.grid(row=3, column=0, pady=5)

        self.rl_scaling_slider = Scale(
            rl_scaling_frame,
            from_=1.0,
            to=10.0,
            orient=tk.HORIZONTAL,
            label="RL Voltage Scaling Factor",
            length=300,
            resolution=0.1,
            command=self.set_rl_scaling_factor
        )
        self.rl_scaling_slider.set(self.rl_scaling_factor)
        self.rl_scaling_slider.pack(padx=5)

        # Move to position input and button
        position_frame = Frame(control_frame)
        position_frame.grid(row=4, column=0, pady=10)

        Label(position_frame, text="Target Position (degrees):").grid(row=0, column=0, padx=5)
        self.position_entry = Entry(position_frame, width=10)
        self.position_entry.grid(row=0, column=1, padx=5)
        self.position_entry.insert(0, "0.0")

        self.move_btn = Button(position_frame, text="Move to Position",
                               command=self.start_move_to_position, width=15)
        self.move_btn.grid(row=0, column=2, padx=5)

        # Stop button
        self.stop_btn = Button(control_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=20, height=2,
                               bg="red", fg="white")
        self.stop_btn.grid(row=5, column=0, pady=10)

        # Manual voltage control
        self.voltage_slider = Scale(
            control_frame,
            from_=-self.max_voltage,
            to=self.max_voltage,
            orient=tk.HORIZONTAL,
            label="Manual Voltage",
            length=300,
            resolution=0.1,
            command=self.set_manual_voltage
        )
        self.voltage_slider.set(0)
        self.voltage_slider.grid(row=6, column=0, padx=5, pady=10)

        # ---------- Training Frame Elements ----------
        Label(training_frame, text="RL Training Settings", font=("Arial", 12, "bold")).grid(row=0, column=0,
                                                                                            columnspan=2, pady=5)

        # Training mode checkbox
        self.training_var = IntVar()
        self.training_checkbox = Checkbutton(
            training_frame,
            text="Enable Training Mode",
            variable=self.training_var,
            command=self.toggle_training_mode
        )
        self.training_checkbox.grid(row=1, column=0, sticky=tk.W, pady=5)

        # Noise slider for exploration
        noise_frame = Frame(training_frame)
        noise_frame.grid(row=2, column=0, pady=5, sticky=tk.W)

        self.noise_slider = Scale(
            noise_frame,
            from_=0.0,
            to=0.5,
            orient=tk.HORIZONTAL,
            label="Exploration Noise",
            length=200,
            resolution=0.01,
            command=self.set_exploration_noise
        )
        self.noise_slider.set(self.exploration_noise)
        self.noise_slider.pack(padx=5)

        # Save model button
        self.save_model_btn = Button(
            training_frame,
            text="Save Current Model",
            command=self.save_current_model,
            width=20
        )
        self.save_model_btn.grid(row=3, column=0, pady=10)

        # Training stats display
        training_stats_frame = Frame(training_frame)
        training_stats_frame.grid(row=4, column=0, pady=5, sticky=tk.W)

        Label(training_stats_frame, text="Training Statistics:").grid(row=0, column=0, sticky=tk.W)
        Label(training_stats_frame, text="Episodes:").grid(row=1, column=0, sticky=tk.W)
        self.episodes_label = Label(training_stats_frame, text="0")
        self.episodes_label.grid(row=1, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Buffer Size:").grid(row=2, column=0, sticky=tk.W)
        self.buffer_label = Label(training_stats_frame, text="0/100000")
        self.buffer_label.grid(row=2, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Updates:").grid(row=3, column=0, sticky=tk.W)
        self.updates_label = Label(training_stats_frame, text="0")
        self.updates_label.grid(row=3, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Avg Reward:").grid(row=4, column=0, sticky=tk.W)
        self.avg_reward_label = Label(training_stats_frame, text="0.0")
        self.avg_reward_label.grid(row=4, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Last Loss:").grid(row=5, column=0, sticky=tk.W)
        self.loss_label = Label(training_stats_frame, text="N/A")
        self.loss_label.grid(row=5, column=1, sticky=tk.W)

        # Add best performance display
        Label(training_stats_frame, text="Best Episode:").grid(row=6, column=0, sticky=tk.W)
        self.best_episode_label = Label(training_stats_frame, text="N/A")
        self.best_episode_label.grid(row=6, column=1, sticky=tk.W)

        # Add curriculum display
        Label(training_stats_frame, text="Curriculum Stage:").grid(row=7, column=0, sticky=tk.W)
        self.curriculum_label = Label(training_stats_frame, text="0")
        self.curriculum_label.grid(row=7, column=1, sticky=tk.W)

        # Auto-save settings
        auto_save_frame = Frame(training_frame)
        auto_save_frame.grid(row=5, column=0, pady=10, sticky=tk.W)

        Label(auto_save_frame, text="Auto-save every:").grid(row=0, column=0, sticky=tk.W)

        self.save_interval_var = StringVar()
        self.save_interval_var.set("100")  # Set to 100 updates (more frequent saving)
        self.save_interval_entry = Entry(auto_save_frame, textvariable=self.save_interval_var, width=6)
        self.save_interval_entry.grid(row=0, column=1, padx=5)

        Label(auto_save_frame, text="updates").grid(row=0, column=2, sticky=tk.W)

        # ---------- Status Display ----------
        Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = Label(status_frame, text="Ready - Please calibrate", width=40)
        self.status_label.grid(row=0, column=1, sticky=tk.W)

        Label(status_frame, text="Model:").grid(row=1, column=0, sticky=tk.W)
        self.model_label = Label(status_frame, text="No model loaded", width=40)
        self.model_label.grid(row=1, column=1, sticky=tk.W)

        Label(status_frame, text="Motor Angle:").grid(row=2, column=0, sticky=tk.W)
        self.angle_label = Label(status_frame, text="0.0°")
        self.angle_label.grid(row=2, column=1, sticky=tk.W)

        Label(status_frame, text="Pendulum Angle:").grid(row=3, column=0, sticky=tk.W)
        self.pendulum_label = Label(status_frame, text="0.0°")
        self.pendulum_label.grid(row=3, column=1, sticky=tk.W)

        Label(status_frame, text="Motor RPM:").grid(row=4, column=0, sticky=tk.W)
        self.rpm_label = Label(status_frame, text="0.0")
        self.rpm_label.grid(row=4, column=1, sticky=tk.W)

        Label(status_frame, text="Current Voltage:").grid(row=5, column=0, sticky=tk.W)
        self.voltage_label = Label(status_frame, text="0.0 V")
        self.voltage_label.grid(row=5, column=1, sticky=tk.W)

        # Training mode indicator
        Label(status_frame, text="Training:").grid(row=6, column=0, sticky=tk.W)
        self.training_label = Label(status_frame, text="OFF", fg="red")
        self.training_label.grid(row=6, column=1, sticky=tk.W)

        # Add balance time indicator
        Label(status_frame, text="Balance Time:").grid(row=7, column=0, sticky=tk.W)
        self.balance_time_label = Label(status_frame, text="0.0s")
        self.balance_time_label.grid(row=7, column=1, sticky=tk.W)

        # RGB Control
        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

    def toggle_training_mode(self):
        """Toggle the training mode on/off with improved exploration strategy"""
        self.training_mode = bool(self.training_var.get())

        if self.training_mode:
            # Enter training mode
            self.actor.train()  # Set network to training mode
            self.critic.train()
            self.training_label.config(text="ON", fg="green")
            self.status_label.config(text="Training mode enabled - collecting experiences")

            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_start_time = time.time()
            self.near_balance_time = 0.0
            self.last_balance_check = time.time()

            # Better exploration strategy - cyclical exploration
            if not hasattr(self, 'exploration_cycle_period'):
                self.exploration_cycle_period = 200  # Episodes per exploration cycle
                self.exploration_base = 0.2
                self.exploration_amplitude = 0.2

            # Calculate exploration based on cycle position
            cycle_position = (self.current_episode % self.exploration_cycle_period) / self.exploration_cycle_period
            self.exploration_noise = self.exploration_base + self.exploration_amplitude * (
                        0.5 - abs(cycle_position - 0.5)) * 2

            self.noise_slider.set(self.exploration_noise)

            # Set orange LED for training mode
            self.r_slider.set(999)
            self.g_slider.set(500)
            self.b_slider.set(0)
        else:
            # Exit training mode
            self.actor.eval()  # Set network to evaluation mode
            self.critic.eval()
            self.training_label.config(text="OFF", fg="red")
            self.status_label.config(text="Training mode disabled")

            # Set blue LED when stopping training
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

            # End current episode and log stats
            if self.current_episode_length > 0:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode += 1

                # Update episode stats
                self.episodes_label.config(text=str(self.current_episode))
                if len(self.episode_rewards) > 0:
                    avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10)
                    self.avg_reward_label.config(text=f"{avg_reward:.1f}")

    def set_exploration_noise(self, value):
        """Set the exploration noise level from slider with better scaling"""
        raw_value = float(value)

        # Non-linear scaling for more fine-grained control at lower values
        if raw_value <= 0.2:
            # Finer control for small values (0-0.2)
            self.exploration_noise = raw_value * 0.5
        else:
            # Regular scaling for larger values
            self.exploration_noise = 0.1 + (raw_value - 0.2) * 0.5

        # Update the exploration base value
        if hasattr(self, 'exploration_base'):
            self.exploration_base = max(0.05, min(0.3, self.exploration_noise))

    def set_rl_scaling_factor(self, value):
        """Set the RL scaling factor from slider"""
        self.rl_scaling_factor = float(value)

    def set_manual_voltage(self, value):
        """Set manual voltage from slider"""
        self.motor_voltage = float(value)
        # Reset any automatic control modes
        self.calibrating = False
        self.moving_to_position = False
        self.rl_mode = False

        if self.training_mode:
            self.toggle_training_mode()  # Turn off training mode
            self.training_var.set(0)  # Update checkbox

        if self.rl_model:
            self.rl_control_btn.config(text="Start RL Control")

    def load_rl_model(self):
        """Open file dialog to select the model file"""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select RL Model File",
            filetypes=(("PyTorch Models", "*.pth"), ("All files", "*.*"))
        )

        if filename:
            try:
                # Load the model
                self.actor.load_state_dict(torch.load(filename, map_location=self.device))
                self.status_label.config(text=f"Model loaded: {os.path.basename(filename)}")
                self.rl_model = filename
                self.model_label.config(text=f"Using: {os.path.basename(filename)}")

                # Enable RL control button
                self.rl_control_btn.config(state=tk.NORMAL)

                # Set blue LED to indicate ready
                self.r_slider.set(0)
                self.g_slider.set(0)
                self.b_slider.set(999)

            except Exception as e:
                self.status_label.config(text=f"Error loading model: {str(e)}")

    def save_current_model(self):
        """Save the current model to a file"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create models directory if it doesn't exist
            os.makedirs("trained_models", exist_ok=True)

            # Define filenames
            actor_filename = f"trained_models/actor_{timestamp}.pth"
            critic_filename = f"trained_models/critic_{timestamp}.pth"

            # Save models
            torch.save(self.actor.state_dict(), actor_filename)
            torch.save(self.critic.state_dict(), critic_filename)

            self.status_label.config(text=f"Model saved: {actor_filename}")
            print(f"Model saved: {actor_filename}")

            return actor_filename
        except Exception as e:
            self.status_label.config(text=f"Error saving model: {str(e)}")
            print(f"Error saving model: {str(e)}")
            return None

    def calibrate(self):
        """Simple calibration - move to corner and set as zero"""
        self.calibrating = True
        self.moving_to_position = False
        self.rl_mode = False
        self.voltage_slider.set(0)  # Reset slider

        # Turn off training if active
        if self.training_mode:
            self.toggle_training_mode()
            self.training_var.set(0)

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
            self.voltage_slider.set(0)  # Reset slider
            self.rl_control_btn.config(text="Stop RL Control")

            if self.training_mode:
                self.status_label.config(text="RL control active with TRAINING")
                # Set purple+orange (pinkish) LED for RL+training
                self.r_slider.set(800)
                self.g_slider.set(400)
                self.b_slider.set(800)
            else:
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

            if self.training_mode:
                self.status_label.config(text="Training mode active - waiting for control")
                # Set orange LED for training mode
                self.r_slider.set(999)
                self.g_slider.set(500)
                self.b_slider.set(0)
            else:
                self.status_label.config(text="RL control stopped")
                # Set blue LED when stopped
                self.r_slider.set(0)
                self.g_slider.set(0)
                self.b_slider.set(999)

    def start_move_to_position(self):
        """Start moving to target position"""
        if not self.calibrating and not self.rl_mode:
            try:
                # Get target position from entry field
                self.target_position = float(self.position_entry.get())

                self.moving_to_position = True
                self.voltage_slider.set(0)  # Reset slider
                self.status_label.config(text=f"Moving to {self.target_position:.1f}°...")

                # Turn off training if active
                if self.training_mode:
                    self.toggle_training_mode()
                    self.training_var.set(0)

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

            self.status_label.config(text="Position reached")
            return

        # Simple proportional control
        kp = 0.02  # Low gain
        self.motor_voltage = kp * position_error

        # Limit voltage for safety
        self.motor_voltage = max(-self.max_voltage, min(self.max_voltage, self.motor_voltage))

    def _get_observation(self):
        """Create observation vector from current state - similar to what's in simulation"""
        # Get current state from hardware
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert all angles to radians for the RL model
        motor_angle = np.radians(motor_angle_deg)

        # For pendulum angle, convert to radians and adjust convention:
        pendulum_angle = np.radians(pendulum_angle_deg)
        # Adjust so that upright is 0
        pendulum_angle_norm = normalize_angle(pendulum_angle + np.pi)

        # Get motor velocity - convert from RPM to rad/s
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
                pendulum_velocity = (current_pendulum_angle_rad - self.prev_pendulum_angle_rl) / dt
            else:
                pendulum_velocity = 0.0
            self.prev_pendulum_angle_rl = current_pendulum_angle_rad
            self.prev_time_rl = current_time

        # Create observation vector with sin/cos values for angles to avoid discontinuities
        obs = np.array([
            np.sin(motor_angle), np.cos(motor_angle),
            np.sin(pendulum_angle_norm), np.cos(pendulum_angle_norm),
            motor_velocity / 10.0,  # Scale velocities as in training
            pendulum_velocity / 10.0
        ])

        return obs, pendulum_angle_norm, motor_angle, motor_velocity, pendulum_velocity

    def _compute_reward(self, alpha_norm, theta, alpha_dot, theta_dot):
        """Improved reward function with better shaping for real hardware"""

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
        downright_alpha = normalize_angle(alpha_norm - np.pi)
        downright_closeness = np.exp(-10.0 * downright_alpha ** 2)
        stability_factor = np.exp(-1.0 * alpha_dot ** 2)
        bonus += -3.0 * downright_closeness * stability_factor  # Smoothly scales based on both factors

        # COMPONENT 5: Smoother penalty for approaching limits
        # Create a continuous penalty that increases as the arm approaches limits
        # Map the distance to limits to a 0-1 range, with 1 being at the limit
        THETA_MIN = -2.2  # Minimum arm angle (radians)
        THETA_MAX = 2.2  # Maximum arm angle (radians)
        theta_max_dist = np.clip(1.0 - abs(theta - THETA_MAX) / 0.5, 0, 1)
        theta_min_dist = np.clip(1.0 - abs(theta - THETA_MIN) / 0.5, 0, 1)
        limit_distance = max(theta_max_dist, theta_min_dist)

        # Apply a nonlinear function to create gradually increasing penalty
        # The penalty grows more rapidly as the arm gets very close to limits
        limit_penalty = -10.0 * limit_distance ** 3

        # COMPONENT 6: Energy management reward
        # This component is already quite smooth, just adjust scaling
        Mp_g_Lp = 0.03013632
        Jp = 0.000131072
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

    def check_episode_termination(self, pendulum_angle_norm, elapsed_time):
        """Check if episode should terminate with more lenient conditions"""
        # More lenient angle threshold - only terminate if completely fallen
        pendulum_fell = abs(pendulum_angle_norm) > np.radians(150)  # Was 160

        # Time-based termination - longer episodes for more learning opportunity
        episode_timeout = elapsed_time > 40.0  # Was 30.0 seconds

        # Success condition - consider episode complete after sustained balancing
        sustained_balance = (
                elapsed_time > 15.0 and  # Must balance for at least 15 seconds
                abs(pendulum_angle_norm) < np.radians(20) and  # Currently near balanced
                self.near_balance_time > 12.0  # Has been near balanced for 12+ seconds
        )

        return pendulum_fell or episode_timeout or sustained_balance

    def update_curriculum(self):
        """Update training curriculum based on performance"""
        if not hasattr(self, 'curriculum_stage'):
            self.curriculum_stage = 0
            self.curriculum_success_counter = 0
            self.curriculum_failure_counter = 0

        # Track successes and failures
        if self.current_episode_length > 10 and self.near_balance_time > 5.0:
            self.curriculum_success_counter += 1
            self.curriculum_failure_counter = max(0, self.curriculum_failure_counter - 1)
        elif self.current_episode_length <= 5:
            self.curriculum_failure_counter += 1
            self.curriculum_success_counter = max(0, self.curriculum_success_counter - 1)

        # Progress to next stage if consistently successful
        if self.curriculum_success_counter >= 10:
            self.curriculum_stage += 1
            self.curriculum_success_counter = 0
            self.curriculum_failure_counter = 0
            print(f"Advancing to curriculum stage {self.curriculum_stage}")

        # Regress to previous stage if consistently failing
        elif self.curriculum_failure_counter >= 20 and self.curriculum_stage > 0:
            self.curriculum_stage -= 1
            self.curriculum_success_counter = 0
            self.curriculum_failure_counter = 0
            print(f"Regressing to curriculum stage {self.curriculum_stage}")

        # Apply curriculum adjustments
        if self.curriculum_stage == 0:
            # Stage 0: Easier balance - more lenient termination, higher rewards
            self.rl_scaling_factor = 4.0
            self.balance_angle_threshold = np.radians(40)  # More lenient balance definition
        elif self.curriculum_stage == 1:
            # Stage 1: Medium difficulty
            self.rl_scaling_factor = 4.0
            self.balance_angle_threshold = np.radians(30)
        else:
            # Stage 2+: Harder challenges
            self.rl_scaling_factor = 4.0
            self.balance_angle_threshold = np.radians(20)  # Stricter balance definition

        # Update slider to match curriculum
        self.rl_scaling_slider.set(self.rl_scaling_factor)

        # Update curriculum display
        self.curriculum_label.config(text=str(self.curriculum_stage))

    def update_rl_control(self):
        """Update RL control logic with optional training using improved methods"""
        # Get current observation
        obs, pendulum_angle_norm, motor_angle, motor_velocity, pendulum_velocity = self._get_observation()

        # Track time spent near balanced position with improved hysteresis
        current_time = time.time()
        dt = current_time - self.last_balance_check
        self.last_balance_check = current_time

        # Better balance tracking with hysteresis
        if not hasattr(self, 'was_near_balance'):
            self.was_near_balance = False

        is_near_balance = abs(pendulum_angle_norm) < self.balance_angle_threshold

        # Only reset balance time when definitely out of balance zone
        if is_near_balance:
            self.near_balance_time += dt
            self.was_near_balance = True
        elif not is_near_balance and abs(pendulum_angle_norm) > self.balance_angle_threshold * 1.2:
            # Only reset if well outside the balance zone (hysteresis)
            self.near_balance_time = 0.0
            self.was_near_balance = False

        # Update balance time display
        self.balance_time_label.config(text=f"{self.near_balance_time:.1f}s")

        # Get action from RL model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)

            if self.training_mode:
                # Sample from policy distribution during training (with added noise)
                action_tensor, _ = self.actor.sample(state_tensor)

                # Add external exploration noise if needed
                if self.exploration_noise > 0:
                    noise = torch.randn_like(action_tensor) * self.exploration_noise
                    action_tensor = torch.clamp(action_tensor + noise, -1.0, 1.0)

                action = action_tensor.cpu().numpy()[0][0]
            else:
                # Use mean action (no exploration) during evaluation
                action_mean, _ = self.actor(state_tensor)
                action = action_mean.cpu().numpy()[0][0]

        # Convert normalized action [-1, 1] to voltage using adjustable scaling factor
        self.motor_voltage = float(action) * self.max_voltage / self.rl_scaling_factor

        # Training mode - collect experiences and update
        if self.training_mode:
            # Calculate reward for current state using improved reward function
            reward = self._compute_reward(
                pendulum_angle_norm,
                motor_angle,
                pendulum_velocity,
                motor_velocity
            )

            # Update episode tracking
            self.current_episode_reward += reward
            self.current_episode_length += 1

            # Check for episode termination with improved conditions
            elapsed_time = time.time() - self.episode_start_time
            done = self.check_episode_termination(pendulum_angle_norm, elapsed_time)

            if done:
                # Log episode stats
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode += 1

                # Update episode stats display
                self.episodes_label.config(text=str(self.current_episode))
                if len(self.episode_rewards) > 0:
                    avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10)
                    self.avg_reward_label.config(text=f"{avg_reward:.1f}")

                # Record best performance and auto-save
                if self.current_episode_reward > self.best_episode_reward:
                    self.best_episode_reward = self.current_episode_reward
                    self.best_episode_length = self.current_episode_length
                    # Update best episode display
                    self.best_episode_label.config(
                        text=f"R:{self.best_episode_reward:.1f} L:{self.best_episode_length}")

                    # Auto-save best model
                    best_model_path = os.path.join("trained_models", f"best_model_{self.current_episode}.pth")
                    torch.save(self.actor.state_dict(), best_model_path)
                    print(
                        f"New best episode! Reward: {self.best_episode_reward:.2f}, Length: {self.best_episode_length}")

                # Check for training stagnation
                if self.current_episode_length <= 2:
                    self.consecutive_short_episodes += 1
                else:
                    self.consecutive_short_episodes = 0

                # If too many short episodes, consider recovery actions
                if self.consecutive_short_episodes > 30:
                    self.consecutive_short_episodes = 0

                    # Recalculate exploration noise based on cycle
                    cycle_position = (
                                                 self.current_episode % self.exploration_cycle_period) / self.exploration_cycle_period
                    new_exploration_noise = self.exploration_base + self.exploration_amplitude * (
                                0.5 - abs(cycle_position - 0.5)) * 2

                    # Make sure it's significantly higher than current
                    old_noise = self.exploration_noise
                    self.exploration_noise = max(0.3, new_exploration_noise * 1.5)
                    self.noise_slider.set(self.exploration_noise)

                    print(
                        f"Training appears stuck - temporarily increasing exploration from {old_noise:.2f} to {self.exploration_noise:.2f}")

                # Reset for next episode
                self.current_episode_reward = 0
                self.current_episode_length = 0
                self.episode_start_time = time.time()
                self.near_balance_time = 0.0

                # Update curriculum
                self.update_curriculum()

                # Indicate episode end
                print(
                    f"Episode {self.current_episode} ended: Reward={self.episode_rewards[-1]:.2f}, Length={self.episode_lengths[-1]}")

            # Store experience in replay buffer if we have a previous state
            if self.prev_state is not None:
                self.replay_buffer.push(
                    self.prev_state,  # State
                    np.array([action]),  # Action
                    reward,  # Reward
                    obs,  # Next state
                    float(done)  # Done flag
                )
                self.experience_counter += 1

            # Store current state and action for next step
            self.prev_state = obs
            self.prev_action = action

            # Update training stats display
            self.buffer_label.config(text=f"{len(self.replay_buffer)}/{self.replay_buffer_size}")

            # Gradually adjust exploration noise according to cyclical schedule
            if len(self.replay_buffer) > BATCH_SIZE * 5:
                # Calculate cycle position for smooth exploration variation
                cycle_position = (self.current_episode % self.exploration_cycle_period) / self.exploration_cycle_period
                target_noise = self.exploration_base + self.exploration_amplitude * (
                            0.5 - abs(cycle_position - 0.5)) * 2

                # Smooth transition to target noise
                self.exploration_noise = 0.95 * self.exploration_noise + 0.05 * target_noise
                self.exploration_noise = max(self.min_exploration, self.exploration_noise)

                # Update slider without triggering the callback
                self.noise_slider.set(self.exploration_noise)

            # Periodic network updates when buffer has enough samples
            if (len(self.replay_buffer) > self.min_buffer_size_for_training and
                    self.experience_counter % self.update_every_n_steps == 0):

                # Perform multiple training updates
                critic_losses = []
                actor_losses = []

                for _ in range(self.training_steps_per_update):
                    losses = self.update_networks()
                    critic_losses.append(losses['critic_loss'])
                    actor_losses.append(losses['actor_loss'])

                # Calculate average losses
                avg_critic_loss = sum(critic_losses) / len(critic_losses)
                avg_actor_loss = sum(actor_losses) / len(actor_losses)

                # Update loss display
                self.loss_label.config(text=f"C:{avg_critic_loss:.4f} A:{avg_actor_loss:.4f}")

                # Increment update counter
                self.update_counter += 1
                self.updates_label.config(text=str(self.update_counter))

                # Auto-save if needed
                try:
                    save_interval = int(self.save_interval_var.get())
                    if self.update_counter % save_interval == 0:
                        model_path = self.save_current_model()
                        if model_path:
                            print(f"Auto-saved model at {model_path}")
                except ValueError:
                    pass  # Invalid save interval

                # Gradually increase training intensity as buffer fills
                if self.update_counter % 100 == 0 and len(self.replay_buffer) > BATCH_SIZE * 20:
                    # More updates per step as training progresses
                    self.training_steps_per_update = min(5, self.training_steps_per_update + 1)
                    print(f"Increased training steps per update to {self.training_steps_per_update}")

        # Update status
        upright_angle = abs(pendulum_angle_norm) * 180 / np.pi
        if upright_angle < 30:
            if self.training_mode:
                self.status_label.config(text=f"Training: Near balance ({upright_angle:.1f}° from upright)")
            else:
                self.status_label.config(text=f"RL control active - Near balance ({upright_angle:.1f}° from upright)")
        else:
            if self.training_mode:
                self.status_label.config(text=f"Training: Swinging ({upright_angle:.1f}° from upright)")
            else:
                self.status_label.config(text=f"RL control active - Swinging ({upright_angle:.1f}° from upright)")

    def update_networks(self):
        """Update actor and critic networks using SAC algorithm with improved techniques"""
        # Set networks to training mode
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

        # Sample batch with adaptive beta for priorities
        beta = self.replay_buffer.get_beta()  # Progressive beta for importance sampling
        sample_result = self.replay_buffer.sample(BATCH_SIZE, beta=beta)

        if len(sample_result) == 7:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = sample_result

            # Convert to tensors
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
            weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        else:
            # Handle the case where the buffer may be empty
            return {'critic_loss': 0.0, 'actor_loss': 0.0}

        # Update critic with gradient clipping
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)

            if AUTO_ENTROPY_TUNING:
                alpha = torch.exp(self.log_alpha).item()
            else:
                alpha = ALPHA

            target_q = target_q - alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * GAMMA * target_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(state_batch, action_batch)

        # Use Huber loss instead of MSE for more robust training
        q1_loss = F.smooth_l1_loss(current_q1 * weights, target_q * weights)
        q2_loss = F.smooth_l1_loss(current_q2 * weights, target_q * weights)
        critic_loss = q1_loss + q2_loss

        # Optimize critic with gradient clipping
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Tighter gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Calculate TD errors for priority updates
        with torch.no_grad():
            td_error1 = torch.abs(current_q1 - target_q)
            td_error2 = torch.abs(current_q2 - target_q)
            td_error = torch.max(td_error1, td_error2).cpu().numpy().flatten()

        # Update priorities in buffer
        new_priorities = td_error + 1e-6  # Add small constant to avoid zero priority
        self.replay_buffer.update_priorities(indices, new_priorities)

        # Update actor less frequently for stability (every 2 updates)
        if self.update_counter % 2 == 0:
            actions, log_probs = self.actor.sample(state_batch)
            q1, q2 = self.critic(state_batch, actions)
            min_q = torch.min(q1, q2)

            if AUTO_ENTROPY_TUNING:
                alpha = torch.exp(self.log_alpha).item()
            else:
                alpha = ALPHA

            # Actor loss (maximize Q - alpha * log_prob)
            actor_loss = (alpha * log_probs - min_q).mean()

            # Optimize actor with gradient clipping
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # Update automatic entropy tuning parameter
            if AUTO_ENTROPY_TUNING:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = torch.exp(self.log_alpha).item()
        else:
            actor_loss = torch.tensor(0.0)

        # Soft update target networks with reduced TAU (slower updates)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss
        }

    def stop_motor(self):
        """Stop the motor and reset control modes"""
        self.calibrating = False
        self.moving_to_position = False
        self.rl_mode = False
        self.motor_voltage = 0.0
        self.voltage_slider.set(0)

        if self.rl_model:
            self.rl_control_btn.config(text="Start RL Control")

        # Set blue LED when stopped
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(999)

        self.status_label.config(text="Motor stopped")

    def update_gui(self):
        """Update the GUI and control the hardware - called continuously"""
        # Update automatic control modes if active
        if self.calibrating:
            self.update_calibration()
        elif self.moving_to_position:
            self.update_position_control()
        elif self.rl_mode:
            self.update_rl_control()

        # Apply the current motor voltage - THIS IS CRITICAL TO DO ON EVERY LOOP!
        self.qube.setMotorVoltage(self.motor_voltage)

        # Apply RGB values
        self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

        # Update display information
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians for normalized angle calculation
        pendulum_angle_rad = np.radians(pendulum_angle_deg)
        pendulum_angle_norm = normalize_angle(pendulum_angle_rad + np.pi)  # For display

        rpm = self.qube.getMotorRPM()

        self.angle_label.config(text=f"{motor_angle_deg:.1f}°")
        self.pendulum_label.config(
            text=f"{pendulum_angle_deg:.1f}° ({abs(pendulum_angle_norm) * 180 / np.pi:.1f}° from upright)")
        self.rpm_label.config(text=f"{rpm:.1f}")
        self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")


def main():
    print("Starting QUBE Controller with Improved RL Training...")
    print("Will set corner position as zero")

    try:
        # Initialize QUBE
        qube = QUBE(COM_PORT, 115200)

        # Initial reset
        qube.resetMotorEncoder()
        qube.resetPendulumEncoder()
        qube.setMotorVoltage(0.0)
        qube.setRGB(0, 0, 999)  # Blue LED to start
        qube.update()

        # Create GUI
        root = tk.Tk()
        app = QUBEControllerWithRLTraining(root, qube)

        # Main loop
        while True:
            qube.update()
            app.update_gui()
            root.update()
            time.sleep(0.01)

    except tk.TclError:
        # Window closed
        pass
    except KeyboardInterrupt:
        # User pressed Ctrl+C
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Final stop attempt
        try:
            qube.setMotorVoltage(0.0)
            qube.update()
            print("Motor stopped")
        except:
            pass


if __name__ == "__main__":
    main()