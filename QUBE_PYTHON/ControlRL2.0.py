import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog, IntVar, Checkbutton, StringVar, OptionMenu, \
    messagebox
from QUBE import QUBE
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import matplotlib.pyplot as plt
from datetime import datetime
import threading

# Update with your COM port
COM_PORT = "COM10"

# Constants from your training code
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
max_voltage = 2.0  # Maximum motor voltage
THETA_MIN = -2.2  # Minimum arm angle (radians)
THETA_MAX = 2.2  # Maximum arm angle (radians)


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


# Critic Network for value function estimation
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


# Replay Buffer for storing experiences
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
        if len(self.buffer) < batch_size:
            return None
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = map(np.array, zip(*[self.buffer[i] for i in batch]))
        return states, actions, rewards, next_states, dones

    def save_to_file(self, filename):
        """Save buffer to file for later use"""
        np.save(filename, self.buffer)
        print(f"Saved {len(self.buffer)} transitions to {filename}")

    def load_from_file(self, filename):
        """Load buffer from file"""
        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True)
            self.buffer = list(data)
            self.position = len(self.buffer) % self.capacity
            print(f"Loaded {len(self.buffer)} transitions from {filename}")

    def __len__(self):
        return len(self.buffer)


# Helper functions
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


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
                action_mean, _ = self.actor(state)
                return action_mean.cpu().numpy()[0]
        else:
            # Sample action with exploration
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size=256):
        # Sample batch from memory
        batch = memory.sample(batch_size)
        if batch is None:
            return None

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

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

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        print(f"Model saved to {filename}_actor.pth and {filename}_critic.pth")

    def load_actor(self, filename):
        self.actor.load_state_dict(torch.load(filename, map_location=self.device))

    def load_critic(self, filename):
        self.critic.load_state_dict(torch.load(filename, map_location=self.device))


class QUBEControllerWithRLTraining:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        master.title("QUBE Controller with RL Training (Enhanced)")

        # Control state
        self.motor_voltage = 0.0
        self.target_position = 0.0
        self.calibrating = False
        self.rl_mode = False
        self.training_mode = False
        self.moving_to_position = False
        self.rl_model = None
        self.max_voltage = 10.0
        self.auto_reset = False
        self.last_update_time = time.time()
        self.recording = False
        self.record_buffer = []
        self.emergency_stopped = False  # Flag to track if emergency stop occurred

        # State tracking
        self.prev_state = None
        self.prev_action = None
        self.prev_time = None
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.train_iterations = 0
        self.current_episode = 0
        self.balanced_time = 0.0
        self.episode_start_time = None
        self.consecutive_bad_states = 0  # Counter for consecutive bad states

        # Episode statistics
        self.episode_rewards = []
        self.training_losses = {'actor': [], 'critic': []}
        self.episode_balanced_times = []
        self.state_history = []
        self.action_history = []
        self.reward_history = []

        # RL parameters
        self.state_dim = 6  # Our observation space
        self.action_dim = 1  # Motor voltage (normalized)
        self.agent = SACAgent(self.state_dim, self.action_dim)
        self.replay_buffer = ReplayBuffer(1000000)  # 1M capacity for more stable learning

        # Training settings
        self.batch_size = 256*32  # Increased batch size
        self.updates_per_step = 1
        self.exploration_prob = 0.2  # Initial exploration rate
        self.train_every_n_steps = 10
        self.save_policy_every_n_episodes = 50
        self.max_episode_steps = 1300  # Extended maximum episode length (around 30s at 86Hz)
        self.warmup_steps = 200  # Warmup steps before enabling termination conditions
        self.consecutive_limit = 50  # Number of consecutive bad states to end episode
        self.termination_angle = 3.0  # ~170 degrees from upright (almost completely fallen)
        self.min_steps_between_episodes = 100  # Min steps before starting new episode
        self.learning_rate = 3e-4  # Learning rate for the agent
        self.auto_reset_time = 15.0  # Time in seconds before auto-reset when balanced
        self.max_motor_angle = 200.0  # Maximum motor angle in degrees before stopping

        # Settings for different difficulty levels
        self.difficulty_presets = {
            'Easy': {
                'exploration': 0.4,
                'termination_angle': 3.0,
                'consecutive_limit': 60,
                'min_steps_between_episodes': 150,
                'warmup_steps': 250,
                'max_motor_angle': 200.0
            },
            'Medium': {
                'exploration': 0.2,
                'termination_angle': 2.8,
                'consecutive_limit': 40,
                'min_steps_between_episodes': 100,
                'warmup_steps': 200,
                'max_motor_angle': 200.0
            },
            'Hard': {
                'exploration': 0.1,
                'termination_angle': 2.6,
                'consecutive_limit': 30,
                'min_steps_between_episodes': 50,
                'warmup_steps': 100,
                'max_motor_angle': 200.0
            }
        }

        # Create GUI elements
        self.create_gui()

    def create_gui(self):
        # Main frame with better spacing
        main_frame = Frame(self.master)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Control frame
        control_frame = Frame(main_frame, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        control_frame.pack(fill=tk.X)

        # Calibrate button
        calibrate_frame = Frame(control_frame)
        calibrate_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.calibrate_btn = Button(calibrate_frame, text="Calibrate (Set Corner as Zero)",
                                    command=self.calibrate,
                                    width=25, height=2)
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)

        # RL Control frame
        rl_frame = Frame(control_frame)
        rl_frame.grid(row=1, column=0, pady=10, sticky=tk.W)

        self.load_model_btn = Button(rl_frame, text="Load RL Model",
                                     command=self.load_rl_model,
                                     width=15)
        self.load_model_btn.grid(row=0, column=0, padx=5)

        self.rl_control_btn = Button(rl_frame, text="Start RL Control",
                                     command=self.toggle_rl_control,
                                     width=15, state=tk.DISABLED)
        self.rl_control_btn.grid(row=0, column=1, padx=5)

        # RL Training frame with reorganized controls
        training_frame = Frame(main_frame, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        training_frame.pack(fill=tk.X, pady=5)

        # Training header
        Label(training_frame, text="Training Controls", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Training enable checkbox
        training_ctrl_frame = Frame(training_frame)
        training_ctrl_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        self.training_var = IntVar()
        self.training_check = Checkbutton(training_ctrl_frame, text="Enable Training",
                                          variable=self.training_var,
                                          command=self.toggle_training)
        self.training_check.grid(row=0, column=0, padx=5, sticky=tk.W)

        # Auto-reset checkbox
        self.auto_reset_var = IntVar()
        self.auto_reset_check = Checkbutton(training_ctrl_frame, text="Auto-Reset When Balanced",
                                            variable=self.auto_reset_var,
                                            command=self.toggle_auto_reset)
        self.auto_reset_check.grid(row=0, column=1, padx=15, sticky=tk.W)

        # Difficulty level selection
        difficulty_frame = Frame(training_frame)
        difficulty_frame.grid(row=2, column=0, sticky=tk.W, pady=5)

        Label(difficulty_frame, text="Difficulty:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.difficulty_var = StringVar()
        self.difficulty_var.set("Medium")  # Default
        difficulty_options = ["Easy", "Medium", "Hard"]
        difficulty_menu = OptionMenu(difficulty_frame, self.difficulty_var, *difficulty_options,
                                     command=self.set_difficulty)
        difficulty_menu.config(width=10)
        difficulty_menu.grid(row=0, column=1, padx=5, sticky=tk.W)

        # Exploration rate
        Label(difficulty_frame, text="Exploration:").grid(row=0, column=2, padx=15, sticky=tk.W)
        self.exploration_slider = Scale(difficulty_frame, from_=0, to=100,
                                        orient=tk.HORIZONTAL, length=150,
                                        command=self.set_exploration)
        self.exploration_slider.set(int(self.exploration_prob * 100))
        self.exploration_slider.grid(row=0, column=3, padx=5, sticky=tk.W)

        # Learning rate
        lr_frame = Frame(training_frame)
        lr_frame.grid(row=3, column=0, sticky=tk.W, pady=5)

        Label(lr_frame, text="Learning Rate:").grid(row=0, column=0, padx=5, sticky=tk.W)
        lr_values = ["0.01", "0.005", "0.001", "0.0005", "0.0001", "0.00005"]
        self.lr_var = StringVar()
        self.lr_var.set("0.0003")  # Default value matches 3e-4
        lr_menu = OptionMenu(lr_frame, self.lr_var, *lr_values, command=self.set_learning_rate)
        lr_menu.config(width=8)
        lr_menu.grid(row=0, column=1, padx=5, sticky=tk.W)

        # Episode management buttons
        episode_frame = Frame(training_frame)
        episode_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        self.save_model_btn = Button(episode_frame, text="Save Model",
                                     command=self.save_model,
                                     width=12, state=tk.DISABLED)
        self.save_model_btn.grid(row=0, column=0, padx=5)

        self.reset_episode_btn = Button(episode_frame, text="Reset Episode",
                                        command=self.reset_episode,
                                        width=12, state=tk.DISABLED)
        self.reset_episode_btn.grid(row=0, column=1, padx=5)

        self.save_buffer_btn = Button(episode_frame, text="Save Buffer",
                                      command=self.save_replay_buffer,
                                      width=12, state=tk.DISABLED)
        self.save_buffer_btn.grid(row=0, column=2, padx=5)

        self.load_buffer_btn = Button(episode_frame, text="Load Buffer",
                                      command=self.load_replay_buffer,
                                      width=12)
        self.load_buffer_btn.grid(row=0, column=3, padx=5)

        # Resume training button (hidden by default)
        self.resume_frame = Frame(training_frame)
        self.resume_training_btn = Button(self.resume_frame, text="RESUME TRAINING",
                                          command=self.resume_training_after_emergency,
                                          width=20, bg="yellow", fg="black", font=("Arial", 10, "bold"))
        self.resume_training_btn.pack(pady=5)

        # Advanced settings expandable section
        self.show_advanced = IntVar()
        self.advanced_check = Checkbutton(training_frame, text="Show Advanced Settings",
                                          variable=self.show_advanced,
                                          command=self.toggle_advanced_settings)
        self.advanced_check.grid(row=5, column=0, sticky=tk.W, pady=5)

        # Advanced settings frame (initially hidden)
        self.advanced_frame = Frame(training_frame)

        # Terminal conditions
        Label(self.advanced_frame, text="Termination Angle (rad):").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.term_angle_entry = Entry(self.advanced_frame, width=6)
        self.term_angle_entry.insert(0, str(self.termination_angle))
        self.term_angle_entry.grid(row=0, column=1, padx=5, sticky=tk.W)

        Label(self.advanced_frame, text="Consecutive Bad States:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.consecutive_entry = Entry(self.advanced_frame, width=6)
        self.consecutive_entry.insert(0, str(self.consecutive_limit))
        self.consecutive_entry.grid(row=1, column=1, padx=5, sticky=tk.W)

        Label(self.advanced_frame, text="Max Motor Angle (deg):").grid(row=2, column=0, padx=5, sticky=tk.W)
        self.max_motor_entry = Entry(self.advanced_frame, width=6)
        self.max_motor_entry.insert(0, str(self.max_motor_angle))
        self.max_motor_entry.grid(row=2, column=1, padx=5, sticky=tk.W)

        Label(self.advanced_frame, text="Warmup Steps:").grid(row=0, column=2, padx=15, sticky=tk.W)
        self.warmup_entry = Entry(self.advanced_frame, width=6)
        self.warmup_entry.insert(0, str(self.warmup_steps))
        self.warmup_entry.grid(row=0, column=3, padx=5, sticky=tk.W)

        Label(self.advanced_frame, text="Min Steps Between Episodes:").grid(row=1, column=2, padx=15, sticky=tk.W)
        self.min_steps_entry = Entry(self.advanced_frame, width=6)
        self.min_steps_entry.insert(0, str(self.min_steps_between_episodes))
        self.min_steps_entry.grid(row=1, column=3, padx=5, sticky=tk.W)

        Button(self.advanced_frame, text="Apply Settings",
               command=self.apply_advanced_settings, width=15).grid(
            row=2, column=0, columnspan=4, pady=10)

        # Arm Position Control frame
        position_frame = Frame(main_frame, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        position_frame.pack(fill=tk.X, pady=5)

        Label(position_frame, text="Arm Position Control", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=5)

        Label(position_frame, text="Target Position (degrees):").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.position_entry = Entry(position_frame, width=8)
        self.position_entry.grid(row=1, column=1, padx=5, sticky=tk.W)
        self.position_entry.insert(0, "0.0")

        self.move_btn = Button(position_frame, text="Move to Position",
                               command=self.start_move_to_position, width=15)
        self.move_btn.grid(row=1, column=2, padx=5, sticky=tk.W)

        # Add quick position buttons
        quick_pos_frame = Frame(position_frame)
        quick_pos_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=tk.W)

        positions = [("0°", "0"), ("45°", "45"), ("90°", "90"), ("-45°", "-45"), ("-90°", "-90")]
        for i, (label, value) in enumerate(positions):
            Button(quick_pos_frame, text=label, width=6,
                   command=lambda v=value: self.quick_set_position(v)).grid(
                row=0, column=i, padx=5)

        # Manual Voltage Control frame
        voltage_frame = Frame(main_frame, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        voltage_frame.pack(fill=tk.X, pady=5)

        Label(voltage_frame, text="Manual Voltage Control", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        self.voltage_slider = Scale(voltage_frame, from_=-self.max_voltage, to=self.max_voltage,
                                    orient=tk.HORIZONTAL, length=400, resolution=0.1,
                                    command=self.set_manual_voltage)
        self.voltage_slider.set(0)
        self.voltage_slider.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        # Emergency stop button
        self.stop_btn = Button(voltage_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=15, height=2,
                               bg="red", fg="white", font=("Arial", 10, "bold"))
        self.stop_btn.grid(row=1, column=1, padx=15, pady=5)

        # Recording control
        record_frame = Frame(main_frame, padx=10, pady=5)
        record_frame.pack(fill=tk.X, pady=5)

        self.record_btn = Button(record_frame, text="Start Recording",
                                 command=self.toggle_recording, width=15)
        self.record_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.record_status = Label(record_frame, text="Not recording", width=20)
        self.record_status.grid(row=0, column=1, padx=5, sticky=tk.W)

        # Status display
        status_frame = Frame(main_frame, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        status_frame.pack(fill=tk.X, pady=5)

        Label(status_frame, text="System Status", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Current status
        Label(status_frame, text="Status:").grid(row=1, column=0, sticky=tk.W)
        self.status_label = Label(status_frame, text="Ready - Please calibrate", width=40, anchor=tk.W)
        self.status_label.grid(row=1, column=1, sticky=tk.W)

        # Model info
        Label(status_frame, text="Model:").grid(row=2, column=0, sticky=tk.W)
        self.model_label = Label(status_frame, text="No model loaded", width=40, anchor=tk.W)
        self.model_label.grid(row=2, column=1, sticky=tk.W)

        # Motor angle
        Label(status_frame, text="Motor Angle:").grid(row=3, column=0, sticky=tk.W)
        self.angle_label = Label(status_frame, text="0.0°", anchor=tk.W)
        self.angle_label.grid(row=3, column=1, sticky=tk.W)

        # Pendulum angle
        Label(status_frame, text="Pendulum Angle:").grid(row=4, column=0, sticky=tk.W)
        self.pendulum_label = Label(status_frame, text="0.0°", anchor=tk.W)
        self.pendulum_label.grid(row=4, column=1, sticky=tk.W)

        # Motor RPM
        Label(status_frame, text="Motor RPM:").grid(row=5, column=0, sticky=tk.W)
        self.rpm_label = Label(status_frame, text="0.0", anchor=tk.W)
        self.rpm_label.grid(row=5, column=1, sticky=tk.W)

        # Current voltage
        Label(status_frame, text="Current Voltage:").grid(row=6, column=0, sticky=tk.W)
        self.voltage_label = Label(status_frame, text="0.0 V", anchor=tk.W)
        self.voltage_label.grid(row=6, column=1, sticky=tk.W)

        # Update rate
        Label(status_frame, text="Update Rate:").grid(row=7, column=0, sticky=tk.W)
        self.update_rate_label = Label(status_frame, text="0 Hz", anchor=tk.W)
        self.update_rate_label.grid(row=7, column=1, sticky=tk.W)

        # Training statistics
        training_stats_frame = Frame(main_frame, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        training_stats_frame.pack(fill=tk.X, pady=5)

        Label(training_stats_frame, text="Training Statistics", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Arrange stats in two columns for better space usage
        col1 = Frame(training_stats_frame)
        col1.grid(row=1, column=0, padx=10, sticky=tk.W)

        col2 = Frame(training_stats_frame)
        col2.grid(row=1, column=1, padx=10, sticky=tk.W)

        # Column 1 stats
        Label(col1, text="Current Episode:").grid(row=0, column=0, sticky=tk.W)
        self.episode_label = Label(col1, text="0", width=10, anchor=tk.W)
        self.episode_label.grid(row=0, column=1, sticky=tk.W)

        Label(col1, text="Episode Steps:").grid(row=1, column=0, sticky=tk.W)
        self.steps_label = Label(col1, text="0", width=10, anchor=tk.W)
        self.steps_label.grid(row=1, column=1, sticky=tk.W)

        Label(col1, text="Episode Reward:").grid(row=2, column=0, sticky=tk.W)
        self.reward_label = Label(col1, text="0.0", width=10, anchor=tk.W)
        self.reward_label.grid(row=2, column=1, sticky=tk.W)

        Label(col1, text="Time Balanced:").grid(row=3, column=0, sticky=tk.W)
        self.balanced_label = Label(col1, text="0.0 s", width=10, anchor=tk.W)
        self.balanced_label.grid(row=3, column=1, sticky=tk.W)

        # Column 2 stats
        Label(col2, text="Training Updates:").grid(row=0, column=0, sticky=tk.W)
        self.updates_label = Label(col2, text="0", width=10, anchor=tk.W)
        self.updates_label.grid(row=0, column=1, sticky=tk.W)

        Label(col2, text="Replay Buffer:").grid(row=1, column=0, sticky=tk.W)
        self.buffer_label = Label(col2, text="0", width=10, anchor=tk.W)
        self.buffer_label.grid(row=1, column=1, sticky=tk.W)

        Label(col2, text="Actor Loss:").grid(row=2, column=0, sticky=tk.W)
        self.actor_loss_label = Label(col2, text="N/A", width=10, anchor=tk.W)
        self.actor_loss_label.grid(row=2, column=1, sticky=tk.W)

        Label(col2, text="Critic Loss:").grid(row=3, column=0, sticky=tk.W)
        self.critic_loss_label = Label(col2, text="N/A", width=10, anchor=tk.W)
        self.critic_loss_label.grid(row=3, column=1, sticky=tk.W)

        # RGB Control
        rgb_frame = Frame(main_frame, padx=10, pady=10)
        rgb_frame.pack(fill=tk.X, pady=5)

        Label(rgb_frame, text="LED Control", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=5)

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red", length=150)
        self.r_slider.grid(row=1, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green", length=150)
        self.g_slider.grid(row=1, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue", length=150)
        self.b_slider.grid(row=1, column=2, padx=5)

        # Set a default color
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(999)  # Blue by default

        # Apply initial difficulty
        self.set_difficulty("Medium")

        # Hide resume training button initially
        self.resume_frame.grid_forget()

    def toggle_advanced_settings(self):
        """Show or hide advanced settings"""
        if self.show_advanced.get():
            self.advanced_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky=tk.W)
        else:
            self.advanced_frame.grid_forget()

    def apply_advanced_settings(self):
        """Apply advanced settings from the input fields"""
        try:
            self.termination_angle = float(self.term_angle_entry.get())
            self.consecutive_limit = int(self.consecutive_entry.get())
            self.warmup_steps = int(self.warmup_entry.get())
            self.min_steps_between_episodes = int(self.min_steps_entry.get())
            self.max_motor_angle = float(self.max_motor_entry.get())
            self.status_label.config(text=f"Advanced settings applied: Motor limit {self.max_motor_angle}°")
        except ValueError:
            self.status_label.config(text="Invalid settings! Enter numbers only.")

    def toggle_auto_reset(self):
        """Toggle auto reset when balanced"""
        self.auto_reset = bool(self.auto_reset_var.get())
        status = "enabled" if self.auto_reset else "disabled"
        self.status_label.config(text=f"Auto-reset when balanced {status}")

    def toggle_recording(self):
        """Toggle state recording for debugging or data collection"""
        self.recording = not self.recording

        if self.recording:
            self.record_btn.config(text="Stop Recording")
            self.record_status.config(text="Recording...", fg="red")
            self.record_buffer = []  # Clear buffer
        else:
            self.record_btn.config(text="Start Recording")
            self.record_status.config(text="Not recording", fg="black")

            # Save recording if we have data
            if len(self.record_buffer) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qube_recording_{timestamp}.npy"
                np.save(filename, self.record_buffer)
                self.record_status.config(text=f"Saved to {filename}", fg="green")

    def set_difficulty(self, difficulty):
        """Apply difficulty preset settings"""
        if difficulty in self.difficulty_presets:
            preset = self.difficulty_presets[difficulty]

            self.exploration_prob = preset['exploration']
            self.exploration_slider.set(int(self.exploration_prob * 100))

            self.termination_angle = preset['termination_angle']
            self.term_angle_entry.delete(0, tk.END)
            self.term_angle_entry.insert(0, str(self.termination_angle))

            self.consecutive_limit = preset['consecutive_limit']
            self.consecutive_entry.delete(0, tk.END)
            self.consecutive_entry.insert(0, str(self.consecutive_limit))

            self.min_steps_between_episodes = preset['min_steps_between_episodes']
            self.min_steps_entry.delete(0, tk.END)
            self.min_steps_entry.insert(0, str(self.min_steps_between_episodes))

            self.warmup_steps = preset['warmup_steps']
            self.warmup_entry.delete(0, tk.END)
            self.warmup_entry.insert(0, str(self.warmup_steps))

            # Apply motor angle limit
            self.max_motor_angle = preset['max_motor_angle']

            self.status_label.config(text=f"Difficulty set to {difficulty}: Motor limit {self.max_motor_angle}°")

    def set_learning_rate(self, lr_str):
        """Set learning rate from dropdown"""
        try:
            lr = float(lr_str)
            self.learning_rate = lr

            # Update optimizer learning rates if agent exists
            for param_group in self.agent.actor_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in self.agent.critic_optimizer.param_groups:
                param_group['lr'] = lr

            self.status_label.config(text=f"Learning rate set to {lr}")
        except ValueError:
            self.status_label.config(text="Invalid learning rate")

    def save_replay_buffer(self):
        """Save the replay buffer to a file"""
        if len(self.replay_buffer) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"replay_buffer_{timestamp}.npy"
            self.replay_buffer.save_to_file(filename)
            self.status_label.config(text=f"Replay buffer saved to {filename}")
        else:
            self.status_label.config(text="Replay buffer is empty")

    def load_replay_buffer(self):
        """Load a replay buffer from a file"""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Replay Buffer File",
            filetypes=(("NumPy Files", "*.npy"), ("All files", "*.*"))
        )

        if filename:
            self.replay_buffer.load_from_file(filename)
            self.buffer_label.config(text=str(len(self.replay_buffer)))
            self.status_label.config(text=f"Loaded buffer with {len(self.replay_buffer)} transitions")

    def quick_set_position(self, value):
        """Quickly set a position from the preset buttons"""
        self.position_entry.delete(0, tk.END)
        self.position_entry.insert(0, value)
        self.start_move_to_position()

    def resume_training_after_emergency(self):
        """Resume training after an emergency stop"""
        # Confirm the user wants to resume
        confirmation = tk.messagebox.askquestion("Confirm Resume Training",
                                                 "Are you sure you want to resume training?\n\n"
                                                 "Make sure you've checked the hardware and resolved any issues.",
                                                 icon='warning')

        if confirmation == 'yes':
            # Hide the resume button
            self.resume_frame.grid_forget()

            # Reset emergency flag
            self.emergency_stopped = False

            # Reset the status color
            self.status_label.config(fg="black")

            # Re-enable training
            self.training_var.set(1)
            self.toggle_training()

            # Start RL control
            self.toggle_rl_control()

            # Reset episode to start fresh
            self.reset_episode()

            self.status_label.config(text="Training resumed after emergency stop")

    def toggle_training(self):
        """Toggle training mode on/off"""
        self.training_mode = bool(self.training_var.get())
        if self.training_mode:
            self.save_model_btn.config(state=tk.NORMAL)
            self.reset_episode_btn.config(state=tk.NORMAL)
            self.save_buffer_btn.config(state=tk.NORMAL)
            self.status_label.config(text="RL training mode enabled")

            # Reset episode stats
            self.reset_episode()
        else:
            self.save_model_btn.config(state=tk.DISABLED)
            self.reset_episode_btn.config(state=tk.DISABLED)
            self.save_buffer_btn.config(state=tk.DISABLED)
            self.status_label.config(text="RL training mode disabled")

    def reset_episode(self):
        """Reset the current episode stats"""
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.balanced_time = 0.0
        self.episode_start_time = time.time()
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.prev_state = None
        self.prev_action = None
        self.consecutive_bad_states = 0

        # Update UI
        self.steps_label.config(text=str(self.episode_steps))
        self.reward_label.config(text=f"{self.episode_reward:.2f}")
        self.balanced_label.config(text=f"{self.balanced_time:.2f} s")
        self.buffer_label.config(text=str(len(self.replay_buffer)))

        # Set LED color based on mode
        if self.rl_mode and self.training_mode:
            # Purple for training
            self.r_slider.set(800)
            self.g_slider.set(0)
            self.b_slider.set(800)
        elif self.rl_mode:
            # Blue for control only
            self.r_slider.set(500)
            self.g_slider.set(0)
            self.b_slider.set(999)

        self.status_label.config(text=f"Episode {self.current_episode} started")

    def set_exploration(self, value):
        """Set exploration probability from slider"""
        self.exploration_prob = float(value) / 100.0

    def save_model(self):
        """Save the current model state"""
        if not self.rl_mode or not self.training_mode:
            self.status_label.config(text="Cannot save: Enable RL + training mode first")
            return

        # Create a timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qube_rl_model_ep{self.current_episode}_{timestamp}"

        # Save the model
        self.agent.save(filename)

        # Also save training history if we have data
        if len(self.episode_rewards) > 0:
            self.save_training_history(filename)

        self.status_label.config(text=f"Model saved as {filename}_actor.pth")

    def save_training_history(self, filename):
        """Save training history as a plot"""
        # Create plot with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot episode rewards
        episodes = range(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards, 'b-')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Progress')
        ax1.grid(True)

        # Plot balanced time
        if len(self.episode_balanced_times) > 0:
            ax2.plot(episodes, self.episode_balanced_times, 'g-')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Balanced Time (s)')
            ax2.set_title('Balancing Performance')
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{filename}_training_history.png")
        plt.close()

    def calibrate(self):
        """Simple calibration - move to corner and set as zero"""
        self.calibrating = True
        self.moving_to_position = False
        self.rl_mode = False
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
                self.agent.load_actor(filename)
                self.status_label.config(text=f"Model loaded: {os.path.basename(filename)}")
                self.rl_model = filename
                self.model_label.config(text=f"Using: {os.path.basename(filename)}")

                # Try to load matching critic if training will be enabled
                critic_file = filename.replace("_actor", "_critic")
                if os.path.exists(critic_file):
                    try:
                        self.agent.load_critic(critic_file)
                        self.status_label.config(text=f"Model and critic loaded")
                    except Exception as e:
                        print(f"Error loading critic: {str(e)}")
                        # Not critical if critic fails to load
                        pass

                # Enable RL control button
                self.rl_control_btn.config(state=tk.NORMAL)

                # Set blue LED to indicate ready
                self.r_slider.set(0)
                self.g_slider.set(0)
                self.b_slider.set(999)

            except Exception as e:
                self.status_label.config(text=f"Error loading model: {str(e)}")
                print(f"Error details: {str(e)}")

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
                self.status_label.config(text="RL control with training active")
                self.reset_episode()
                # Set purple LED during RL training
                self.r_slider.set(800)
                self.g_slider.set(0)
                self.b_slider.set(800)
            else:
                self.status_label.config(text="RL control active")
                # Set cyan LED during RL control
                self.r_slider.set(0)
                self.g_slider.set(500)
                self.b_slider.set(999)

        else:
            # Stop RL control
            self.rl_mode = False
            self.motor_voltage = 0.0
            self.rl_control_btn.config(text="Start RL Control")

            if self.training_mode:
                # If we were training, save episode results
                self.finish_episode()

            self.status_label.config(text="RL control stopped")

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def finish_episode(self):
        """Finish the current episode and record statistics"""
        if self.episode_steps > 0:
            # Record episode stats
            self.episode_rewards.append(self.episode_reward)
            self.episode_balanced_times.append(self.balanced_time)
            self.current_episode += 1

            # Log results
            print(f"Episode {self.current_episode} completed: " +
                  f"Steps={self.episode_steps}, " +
                  f"Reward={self.episode_reward:.2f}, " +
                  f"Balanced={self.balanced_time:.2f}s")

            # Auto-save policy if needed
            if self.current_episode % self.save_policy_every_n_episodes == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qube_rl_autosave_ep{self.current_episode}_{timestamp}"
                self.agent.save(filename)
                print(f"Auto-saved model at episode {self.current_episode}")

            # Reset for next episode
            self.reset_episode()

    def get_state_observation(self, motor_angle_deg, pendulum_angle_deg, motor_velocity, pendulum_velocity):
        """Convert raw sensor values to normalized observation for RL"""
        # Convert angles to radians
        motor_angle = np.radians(motor_angle_deg)

        # Convert pendulum angle to radians and normalize
        pendulum_angle = np.radians(pendulum_angle_deg)
        pendulum_angle_norm = normalize_angle(pendulum_angle + np.pi)

        # Create observation vector (same format as in training)
        obs = np.array([
            np.sin(motor_angle), np.cos(motor_angle),
            np.sin(pendulum_angle_norm), np.cos(pendulum_angle_norm),
            motor_velocity / 10.0,  # Scale velocities as in training
            pendulum_velocity / 10.0
        ])

        return obs

    def compute_reward(self, motor_angle_deg, pendulum_angle_deg, motor_velocity, pendulum_velocity, voltage):
        """Compute reward based on the current state (similar to simulator reward)"""
        # Convert to radians
        motor_angle = np.radians(motor_angle_deg)
        pendulum_angle = np.radians(pendulum_angle_deg)
        pendulum_angle_norm = normalize_angle(pendulum_angle + np.pi)

        # Component 1: Reward for pendulum being close to upright
        upright_reward = np.cos(pendulum_angle_norm)  # 1 when upright, -1 when downward

        # COMPONENT 2: Smooth penalty for high velocities - quadratic falloff
        # Use tanh to create a smoother penalty that doesn't grow excessively large
        velocity_norm = (motor_velocity ** 2 + pendulum_velocity ** 2) / 10.0  # Normalize velocities
        velocity_penalty = -0.3 * np.tanh(velocity_norm)  # Bounded penalty

        # Component 3: Penalty for arm position away from center
        pos_penalty = -0.1 * np.tanh(motor_angle ** 2 / 2.0)

        # COMPONENT 4: Smoother bonus for being close to upright position
        upright_closeness = np.exp(-10.0 * pendulum_angle_norm ** 2)  # Close to 1 when near upright, falls off quickly
        stability_factor = np.exp(-1.0 * pendulum_velocity ** 2)  # Close to 1 when velocity is low
        bonus = 3.0 * upright_closeness * stability_factor  # Smoothly scales based on both factors

        # COMPONENT 4.5: Smoother cost for being close to downright position
        # For new convention, downright is at π
        downright_alpha = normalize_angle(pendulum_angle - np.pi)
        downright_closeness = np.exp(-10.0 * downright_alpha ** 2)
        stability_factor = np.exp(-1.0 * pendulum_velocity ** 2)
        bonus += -3.0 * downright_closeness * stability_factor  # Smoothly scales based on both factors


        # Component 4: Extra reward for being very close to upright and stable
        bonus = 0.0
        if abs(pendulum_angle_norm) < 0.2 and abs(pendulum_velocity) < 1.0:
            bonus = 5.0

        # COMPONENT 5: Smoother penalty for approaching limits
        # Create a continuous penalty that increases as the arm approaches limits
        # Map the distance to limits to a 0-1 range, with 1 being at the limit
        theta_max_dist = np.clip(1.0 - abs(motor_angle - THETA_MAX) / 0.5, 0, 1)
        theta_min_dist = np.clip(1.0 - abs(motor_angle - THETA_MIN) / 0.5, 0, 1)
        limit_distance = max(theta_max_dist, theta_min_dist)

        # Apply a nonlinear function to create gradually increasing penalty
        # The penalty grows more rapidly as the arm gets very close to limits
        limit_penalty = -10.0 * limit_distance ** 3

        # Component 6: Energy management reward
        Mp_g_Lp = mL * g * LL  # Pendulum mass * gravity * length
        Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia
        energy_reward = 2 - 0.15 * abs(Mp_g_Lp * (1 - np.cos(pendulum_angle_norm))
                                + 0.5 * Jp * pendulum_velocity ** 2
                                - Mp_g_Lp)  # Difference from optimal energy


        # Combine all components
        reward = (
                upright_reward
                # + velocity_penalty
                + pos_penalty
                + bonus
                + limit_penalty
                + energy_reward
        )

        # Check if pendulum is balanced (for tracking balanced time)
        is_balanced = abs(pendulum_angle_norm) < 0.17  # ~10 degrees

        return reward, is_balanced

    def start_move_to_position(self):
        """Start moving to target position"""
        if not self.calibrating and not self.rl_mode:
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

            self.status_label.config(text="Position reached")
            return

        # Simple proportional control with improved gain
        kp = 0.05  # Increased gain for faster movement

        # Get motor velocity to add damping
        motor_velocity = self.qube.getMotorRPM() * (2 * np.pi / 60)
        kd = 0.02  # Damping coefficient

        # PD control
        self.motor_voltage = kp * position_error - kd * motor_velocity

        # Limit voltage for safety
        self.motor_voltage = max(-self.max_voltage, min(self.max_voltage, self.motor_voltage))

    def check_auto_reset(self, pendulum_angle_norm):
        """Check if the pendulum should be auto-reset after balancing for a while"""
        if self.auto_reset and self.episode_steps > self.warmup_steps and abs(pendulum_angle_norm) < 0.17:
            elapsed_time = time.time() - self.episode_start_time
            if elapsed_time > self.auto_reset_time:
                self.status_label.config(text="Auto-resetting after successful balance")
                self.finish_episode()
                self.reset_episode()
                return True
        return False

    def update_rl_control(self):
        """Update RL control and training logic"""
        # Get current state
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()
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

        # Get observation for RL
        obs = self.get_state_observation(motor_angle_deg, pendulum_angle_deg,
                                         motor_velocity, pendulum_velocity)

        # Record state if recording is enabled
        if self.recording:
            self.record_buffer.append([
                current_time,
                motor_angle_deg,
                pendulum_angle_deg,
                motor_velocity,
                pendulum_velocity,
            ])

        # Decide action based on the current policy
        if self.training_mode and np.random.random() < self.exploration_prob:
            # Exploration: take random action
            action = np.random.uniform(-1, 1, size=(1,))
        else:
            # Exploitation: use policy
            action = self.agent.select_action(obs, evaluate=not self.training_mode)

        # Apply action as voltage
        self.motor_voltage = float(action[0]) / 5.0 * self.max_voltage

        # If training mode is active, handle training logic
        if self.training_mode:
            # Compute reward
            reward, is_balanced = self.compute_reward(
                motor_angle_deg, pendulum_angle_deg,
                motor_velocity, pendulum_velocity,
                action[0]
            )

            # Compute normalized pendulum angle for termination check
            pendulum_angle_norm = normalize_angle(current_pendulum_angle_rad + np.pi)

            # Check if the pendulum has fallen too far (should end episode)
            if abs(pendulum_angle_norm) > self.termination_angle:
                self.consecutive_bad_states += 1
            else:
                self.consecutive_bad_states = 0

            # Update balanced time if pendulum is balanced
            if is_balanced:
                self.balanced_time += current_time - self.prev_time_rl if self.prev_time_rl else 0

            # Update UI
            self.balanced_label.config(text=f"{self.balanced_time:.2f} s")

            # Check auto-reset condition
            auto_reset_happened = self.check_auto_reset(pendulum_angle_norm)

            # If we have a previous state, store transition
            if self.prev_state is not None and not auto_reset_happened:
                # Check if episode should end, but only after warmup period
                done = False

                # End if step limit reached
                if self.episode_steps >= self.max_episode_steps:
                    done = True
                    print("Episode ended due to step limit")

                # End if pendulum has fallen beyond recovery for consecutive steps
                # But only apply this after warmup period
                if self.episode_steps > self.warmup_steps and self.consecutive_bad_states > self.consecutive_limit:
                    done = True
                    print(
                        f"Episode ended due to pendulum fall: {abs(pendulum_angle_norm)} rad, {self.consecutive_bad_states} consecutive bad states")

                # End if motor angle exceeds 200 degrees - pendulum is likely off the motor
                if abs(motor_angle_deg) > 200.0:
                    done = True
                    print(
                        f"Episode ended because motor angle ({motor_angle_deg:.1f}°) exceeds 200° - pendulum arm is likely off motor")
                    self.motor_voltage = 0.0  # Immediately stop the motor

                    # Also exit training mode entirely
                    self.status_label.config(
                        text=f"TRAINING STOPPED: Motor angle ({motor_angle_deg:.1f}°) exceeded 200°")
                    # Schedule end of RL mode to avoid threading issues
                    self.master.after(100, self.stop_training_emergency)

                # End if motor angle exceeds safety limit
                if abs(motor_angle_deg) > self.max_motor_angle:
                    done = True
                    print(
                        f"Episode ended due to excessive motor angle: {motor_angle_deg:.1f}° exceeds limit of {self.max_motor_angle:.1f}°")
                    # Set motor voltage to 0 immediately for safety
                    self.motor_voltage = 0.0

                # Store the transition
                self.replay_buffer.push(
                    self.prev_state,
                    self.prev_action,
                    reward,
                    obs,
                    done
                )

                # Accumulate episode reward
                self.episode_reward += reward

                # Store history for plotting
                self.state_history.append(
                    [motor_angle_deg, pendulum_angle_deg, motor_velocity, pendulum_velocity]
                )
                self.action_history.append(action[0])
                self.reward_history.append(reward)

                # Update model periodically
                if self.episode_steps % self.train_every_n_steps == 0 and len(self.replay_buffer) >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        losses = self.agent.update_parameters(self.replay_buffer, self.batch_size)
                        if losses is not None:
                            self.training_losses['actor'].append(losses['actor_loss'])
                            self.training_losses['critic'].append(losses['critic_loss'])

                            # Update UI
                            self.actor_loss_label.config(text=f"{losses['actor_loss']:.4f}")
                            self.critic_loss_label.config(text=f"{losses['critic_loss']:.4f}")

                    self.train_iterations += 1
                    self.updates_label.config(text=str(self.train_iterations))

                # Check if episode is done
                if done:
                    self.finish_episode()

            # Store current state for next step
            self.prev_state = obs.copy()
            self.prev_action = action.copy()

            # Increment step counter
            self.episode_steps += 1
            self.steps_label.config(text=str(self.episode_steps))
            self.reward_label.config(text=f"{self.episode_reward:.2f}")
            self.buffer_label.config(text=str(len(self.replay_buffer)))

        # Update status message
        pendulum_angle_norm = normalize_angle(current_pendulum_angle_rad + np.pi)
        upright_angle_deg = abs(pendulum_angle_norm) * 180 / np.pi

        if self.training_mode:
            msg = f"RL training, Episode {self.current_episode}, Step {self.episode_steps}"
            if self.episode_steps <= self.warmup_steps:
                msg += f" (Warmup: {self.warmup_steps - self.episode_steps} left)"
            if upright_angle_deg < 30:
                msg += f" (Near balance: {upright_angle_deg:.1f}°)"
        else:
            if upright_angle_deg < 30:
                msg = f"RL control - Near balance ({upright_angle_deg:.1f}° from upright)"
            else:
                msg = f"RL control - Swinging ({upright_angle_deg:.1f}° from upright)"

        self.status_label.config(text=msg)

    def set_manual_voltage(self, value):
        """Set manual voltage from slider"""
        self.motor_voltage = float(value)
        # Reset any automatic control modes
        self.calibrating = False
        self.moving_to_position = False
        self.rl_mode = False

        if self.rl_model:
            self.rl_control_btn.config(text="Start RL Control")

    def stop_training_emergency(self):
        """Emergency stop for training when motor angle exceeds limits"""
        # Exit training mode
        self.training_var.set(0)  # Uncheck the training checkbox
        self.toggle_training()  # Apply the change

        # Exit RL control mode
        self.rl_mode = False
        self.rl_control_btn.config(text="Start RL Control")

        # Set red alert LED
        self.r_slider.set(999)
        self.g_slider.set(0)
        self.b_slider.set(0)

        # Keep motor voltage at 0
        self.motor_voltage = 0.0
        self.voltage_slider.set(0)

        # Show alert message
        self.status_label.config(text="⚠️ EMERGENCY STOP: Training halted due to excessive motor angle", fg="red")

        # Set emergency flag and show resume button
        self.emergency_stopped = True
        self.resume_frame.grid(row=5, column=0, columnspan=2, pady=10)

        # Create an alert popup
        tk.messagebox.showwarning("Training Emergency Stop",
                                  "Training has been stopped because the motor angle exceeded 200°.\n\n"
                                  "This usually means the pendulum came off the motor or hit something.\n\n"
                                  "Please check the hardware before resuming.")

    def stop_motor(self):
        """Stop the motor"""
        self.calibrating = False
        self.moving_to_position = False

        if self.rl_mode and self.training_mode:
            # Finish the current episode if training
            self.finish_episode()

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
        # Track update rate
        current_time = time.time()
        dt = current_time - self.last_update_time
        if dt > 0:
            update_rate = 1.0 / dt
            self.update_rate_label.config(text=f"{update_rate:.1f} Hz")
        self.last_update_time = current_time

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

        # Check if motor angle is getting close to limit (within 20%)
        angle_warning = abs(motor_angle_deg) > (self.max_motor_angle * 0.8)

        # Update angle display with warning if needed
        if angle_warning:
            self.angle_label.config(text=f"{motor_angle_deg:.1f}° !", fg="red")
        else:
            self.angle_label.config(text=f"{motor_angle_deg:.1f}°", fg="black")

        self.pendulum_label.config(
            text=f"{pendulum_angle_deg:.1f}° ({abs(pendulum_angle_norm) * 180 / np.pi:.1f}° from upright)")
        self.rpm_label.config(text=f"{rpm:.1f}")
        self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")

        # Force the master widget to update
        self.master.update_idletasks()


def main():
    print("Starting QUBE Controller with Enhanced RL Training...")
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
            try:
                qube.update()
                app.update_gui()
                root.update()
                time.sleep(0.005)  # Shorter sleep for faster update rate
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                # Continue despite errors to keep the controller running

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