import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog, IntVar, Checkbutton
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
max_voltage = 10.0  # Maximum motor voltage
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
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = map(np.array, zip(*[self.buffer[i] for i in batch]))
        return states, actions, rewards, next_states, dones

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
        if len(memory) < batch_size:
            return None

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

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load_actor(self, filename):
        self.actor.load_state_dict(torch.load(filename, map_location=self.device))

    def load_critic(self, filename):
        self.critic.load_state_dict(torch.load(filename, map_location=self.device))


class QUBEControllerWithRLTraining:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        master.title("QUBE Controller with RL Training")

        # Control state
        self.motor_voltage = 0.0
        self.target_position = 0.0
        self.calibrating = False
        self.rl_mode = False
        self.training_mode = False
        self.moving_to_position = False
        self.rl_model = None
        self.max_voltage = 10.0

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
        self.replay_buffer = ReplayBuffer(100000)  # 100k capacity

        # Training settings
        self.batch_size = 256
        self.updates_per_step = 1
        self.exploration_prob = 0.2  # Probability of exploration
        self.train_every_n_steps = 10  # Update model every n steps
        self.save_policy_every_n_episodes = 5  # Save policy every n episodes
        self.max_episode_steps = 1300  # Maximum steps per episode (15s at ~86Hz)

        # Create GUI elements
        self.create_gui()

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

        # RL Training controls
        training_frame = Frame(control_frame)
        training_frame.grid(row=2, column=0, pady=10)

        self.training_var = IntVar()
        self.training_check = Checkbutton(training_frame, text="Enable Training",
                                          variable=self.training_var,
                                          command=self.toggle_training)
        self.training_check.grid(row=0, column=0, padx=5)

        Label(training_frame, text="Exploration:").grid(row=0, column=1, padx=5)
        self.exploration_slider = Scale(training_frame, from_=0, to=100,
                                        orient=tk.HORIZONTAL, length=200,
                                        command=self.set_exploration)
        self.exploration_slider.set(int(self.exploration_prob * 100))
        self.exploration_slider.grid(row=0, column=2, padx=5)

        save_frame = Frame(training_frame)
        save_frame.grid(row=1, column=0, columnspan=3, pady=5)

        self.save_model_btn = Button(save_frame, text="Save Current Model",
                                     command=self.save_model,
                                     width=20, state=tk.DISABLED)
        self.save_model_btn.grid(row=0, column=0, padx=5)

        self.reset_episode_btn = Button(save_frame, text="Reset Episode",
                                        command=self.reset_episode,
                                        width=15, state=tk.DISABLED)
        self.reset_episode_btn.grid(row=0, column=1, padx=5)

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

        # Stop button
        self.stop_btn = Button(control_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=20, height=2,
                               bg="red", fg="white")
        self.stop_btn.grid(row=4, column=0, pady=10)

        # Manual voltage control
        self.voltage_slider = Scale(
            control_frame,
            from_=-self.max_voltage,
            to=self.max_voltage,
            orient=tk.HORIZONTAL,
            label="Manual Voltage",
            length=400,
            resolution=0.1,
            command=self.set_manual_voltage
        )
        self.voltage_slider.set(0)
        self.voltage_slider.grid(row=5, column=0, padx=5, pady=10)

        # Status display
        status_frame = Frame(self.master, padx=10, pady=10)
        status_frame.pack()

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

        # Training status
        training_stats_frame = Frame(self.master, padx=10, pady=10)
        training_stats_frame.pack()

        Label(training_stats_frame, text="Training Status", font=("Arial", 12, "bold")).grid(row=0, column=0,
                                                                                             columnspan=2, pady=5)

        Label(training_stats_frame, text="Current Episode:").grid(row=1, column=0, sticky=tk.W)
        self.episode_label = Label(training_stats_frame, text="0")
        self.episode_label.grid(row=1, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Episode Steps:").grid(row=2, column=0, sticky=tk.W)
        self.steps_label = Label(training_stats_frame, text="0")
        self.steps_label.grid(row=2, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Episode Reward:").grid(row=3, column=0, sticky=tk.W)
        self.reward_label = Label(training_stats_frame, text="0.0")
        self.reward_label.grid(row=3, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Time Balanced:").grid(row=4, column=0, sticky=tk.W)
        self.balanced_label = Label(training_stats_frame, text="0.0 s")
        self.balanced_label.grid(row=4, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Training Updates:").grid(row=5, column=0, sticky=tk.W)
        self.updates_label = Label(training_stats_frame, text="0")
        self.updates_label.grid(row=5, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Actor Loss:").grid(row=6, column=0, sticky=tk.W)
        self.actor_loss_label = Label(training_stats_frame, text="N/A")
        self.actor_loss_label.grid(row=6, column=1, sticky=tk.W)

        Label(training_stats_frame, text="Critic Loss:").grid(row=7, column=0, sticky=tk.W)
        self.critic_loss_label = Label(training_stats_frame, text="N/A")
        self.critic_loss_label.grid(row=7, column=1, sticky=tk.W)

        # RGB Control
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack()

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

    def toggle_training(self):
        """Toggle training mode on/off"""
        self.training_mode = bool(self.training_var.get())
        if self.training_mode:
            self.save_model_btn.config(state=tk.NORMAL)
            self.reset_episode_btn.config(state=tk.NORMAL)
            self.status_label.config(text="RL training mode enabled")

            # Reset episode stats
            self.reset_episode()
        else:
            self.save_model_btn.config(state=tk.DISABLED)
            self.reset_episode_btn.config(state=tk.DISABLED)
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

        # Update UI
        self.steps_label.config(text=str(self.episode_steps))
        self.reward_label.config(text=f"{self.episode_reward:.2f}")
        self.balanced_label.config(text=f"{self.balanced_time:.2f} s")

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
                    except:
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

        # Component 2: Penalty for high velocities
        velocity_penalty = -0.01 * (motor_velocity ** 2 + pendulum_velocity ** 2)

        # Component 3: Penalty for arm position away from center
        pos_penalty = -0.01 * motor_angle ** 2

        # Component 4: Extra reward for being very close to upright and stable
        bonus = 0.0
        if abs(pendulum_angle_norm) < 0.2 and abs(pendulum_velocity) < 1.0:
            bonus = 5.0

        # Component 5: Penalty for voltage oscillations
        voltage_penalty = -0.01 * voltage ** 2

        # Component 6: Energy management reward
        Mp_g_Lp = mL * g * LL  # Pendulum mass * gravity * length
        Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia

        # Calculate energy
        E = Mp_g_Lp * (1 - np.cos(pendulum_angle_norm)) + 0.5 * Jp * pendulum_velocity ** 2  # Current energy
        E_ref = Mp_g_Lp  # Energy at upright position (target energy)
        E_diff = abs(E - E_ref)  # Difference from optimal energy

        # Reward for being close to the optimal energy (inverted Gaussian)
        energy_reward = 2.0 * np.exp(-0.5 * (E_diff / (0.2 * E_ref)) ** 2)

        # Total reward
        reward = upright_reward + pos_penalty + bonus + energy_reward

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

        # Simple proportional control
        kp = 0.02  # Low gain
        self.motor_voltage = kp * position_error

        # Limit voltage for safety
        self.motor_voltage = max(-self.max_voltage, min(self.max_voltage, self.motor_voltage))

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

        # Decide action based on the current policy
        if self.training_mode and np.random.random() < self.exploration_prob:
            # Exploration: take random action
            action = np.random.uniform(-1, 1, size=(1,))
        else:
            # Exploitation: use policy
            action = self.agent.select_action(obs, evaluate=not self.training_mode)

        # Apply action as voltage
        self.motor_voltage = float(action[0]) * self.max_voltage

        # If training mode is active, handle training logic
        if self.training_mode:
            # Compute reward
            reward, is_balanced = self.compute_reward(
                motor_angle_deg, pendulum_angle_deg,
                motor_velocity, pendulum_velocity,
                action[0]
            )

            # Update balanced time if pendulum is balanced
            if is_balanced:
                self.balanced_time += current_time - self.prev_time_rl if self.prev_time_rl else 0

            # Update UI
            self.balanced_label.config(text=f"{self.balanced_time:.2f} s")

            # If we have a previous state, store transition
            if self.prev_state is not None:
                # Check if episode should end
                done = False

                # End if step limit reached
                if self.episode_steps >= self.max_episode_steps:
                    done = True

                # End if pendulum fell beyond recovery
                pendulum_angle_norm = normalize_angle(current_pendulum_angle_rad + np.pi)
                if abs(pendulum_angle_norm) > 2.8:  # ~160 degrees from upright
                    done = True

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

        # Update status message
        pendulum_angle_norm = normalize_angle(current_pendulum_angle_rad + np.pi)
        upright_angle_deg = abs(pendulum_angle_norm) * 180 / np.pi

        if self.training_mode:
            msg = f"RL training active - Episode {self.current_episode}, Step {self.episode_steps}"
            if upright_angle_deg < 30:
                msg += f" (Near balance: {upright_angle_deg:.1f}°)"
        else:
            if upright_angle_deg < 30:
                msg = f"RL control active - Near balance ({upright_angle_deg:.1f}° from upright)"
            else:
                msg = f"RL control active - Swinging ({upright_angle_deg:.1f}° from upright)"

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
    print("Starting QUBE Controller with RL Training...")
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