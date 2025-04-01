import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for matplotlib


# Helper function to normalize angle to [-π, π]
def normalize_angle(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


# Class definition for simulation parameters
class Params:
    def __init__(self):
        self.Mp_g_Lp = 9.8  # Assuming mass * g * length for pendulum
        self.Jp = 1.0  # Pendulum's moment of inertia
        self.THETA_MAX = np.pi / 2  # Maximum arm angle
        self.THETA_MIN = -np.pi / 2  # Minimum arm angle


# Reward function - MODIFIED FOR UPRIGHT = 0
def compute_reward(state, voltage_change=0.0, params=None):
    """Calculate reward based on current state with smoother transitions"""
    theta, alpha, theta_dot, alpha_dot = state
    # No π shift needed since we want α=0 to be upright
    alpha_norm = normalize_angle(alpha)
    p = params if params else Params()

    # COMPONENT 1: Base reward for pendulum being upright (range: -1 to 1)
    upright_reward = 1.0 * np.cos(alpha_norm)

    # COMPONENT 2: Smooth penalty for high velocities - quadratic falloff
    velocity_norm = (theta_dot ** 2 + alpha_dot ** 2) / 10.0
    velocity_penalty = -0.0 * np.tanh(velocity_norm)

    # COMPONENT 3: Smooth penalty for arm position away from center
    pos_penalty = 3.0 * np.cos(normalize_angle(theta)) - 1.0

    # COMPONENT 4: Smoother bonus for being close to upright position
    upright_closeness = np.exp(-10.0 * alpha_norm ** 2)
    stability_factor = np.exp(-0.1 * alpha_dot ** 2)
    bonus = 3.0 * upright_closeness * stability_factor

    # COMPONENT 4.5: Smoother cost for being close to downright position
    # For new convention, downright is at π
    downright_alpha = normalize_angle(alpha - np.pi)
    downright_closeness = np.exp(-10.0 * downright_alpha ** 2)
    stability_factor = np.exp(-0.1 * alpha_dot ** 2)
    bonus += -3.0 * downright_closeness * stability_factor

    # COMPONENT 5: Smoother penalty for approaching limits
    THETA_MAX = p.THETA_MAX
    THETA_MIN = p.THETA_MIN
    theta_max_dist = np.clip(1.0 - abs(theta - THETA_MAX) / 0.5, 0, 1)
    theta_min_dist = np.clip(1.0 - abs(theta - THETA_MIN) / 0.5, 0, 1)
    limit_distance = max(theta_max_dist, theta_min_dist)
    limit_penalty = -10.0 * limit_distance ** 3

    # COMPONENT 6: Energy management reward
    E = p.Mp_g_Lp * (np.cos(alpha_norm)) + 0.5 * p.Jp * alpha_dot ** 2
    E_ref = p.Mp_g_Lp
    energy_reward = 2 - 0.15 * abs(E_ref - E)

    # COMPONENT 7: Stronger penalty for fast voltage changes
    voltage_change_penalty = -0.01 * voltage_change ** 2

    # Individual components
    components = {
        "Upright Reward": upright_reward,
        "Velocity Penalty": velocity_penalty,
        "Position Penalty": pos_penalty,
        "Upright Bonus": bonus,
        "Limit Penalty": limit_penalty,
        "Energy Reward": energy_reward,
        "Voltage Change": voltage_change_penalty
    }

    # Total reward
    total_reward = (
            upright_reward
            + velocity_penalty
            + pos_penalty
            + bonus
            + limit_penalty
            + energy_reward
            + voltage_change_penalty
    )

    return total_reward, components


# Function to compute individual reward components for plots
def compute_component(component_name, state_grid, voltage_change=0.0, params=None):
    component_map = {
        "Upright Reward": "Upright Reward",
        "Velocity Penalty": "Velocity Penalty",
        "Position Penalty": "Position Penalty",
        "Upright Bonus": "Upright Bonus",
        "Limit Penalty": "Limit Penalty",
        "Energy Reward": "Energy Reward",
        "Voltage Change": "Voltage Change"
    }

    rewards = np.zeros(state_grid[0].shape)
    for i in range(state_grid[0].shape[0]):
        for j in range(state_grid[0].shape[1]):
            state = [state_grid[0][i, j], state_grid[1][i, j], state_grid[2][i, j], state_grid[3][i, j]]
            _, components = compute_reward(state, voltage_change, params)
            rewards[i, j] = components[component_map[component_name]]
    return rewards


class RewardExplorerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Reward Function Explorer")
        self.root.geometry("1200x800")

        # Initialize params
        self.params = Params()

        # Create the main notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create tabs
        self.create_interactive_explorer_tab()
        self.create_reward_vs_alpha_tab()
        self.create_heatmap_tab()
        self.create_surface_plot_tab()
        self.create_component_comparison_tab()
        self.create_swingup_trajectory_tab()  # Add the new tab

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def on_tab_change(self, event):
        # Get the currently selected tab
        tab = self.notebook.tab(self.notebook.select(), "text")
        self.status_var.set(f"Viewing: {tab}")

        # Update the active plot if needed
        if tab == "Reward vs Alpha":
            self.plot_reward_vs_alpha()
        elif tab == "Reward Heatmap":
            self.plot_reward_heatmap()
        elif tab == "3D Surface":
            self.plot_reward_surface()
        elif tab == "Component Comparison":
            self.plot_component_comparison()
        elif tab == "Swing-Up Trajectory":
            self.plot_swingup_trajectory()

    def create_interactive_explorer_tab(self):
        # Create main frame for this tab
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Interactive Explorer")

        # Split into left (controls) and right (visualization) panes
        control_frame = ttk.Frame(tab)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        viz_frame = ttk.Frame(tab)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create matplotlib figure for visualization
        self.explorer_fig = Figure(figsize=(8, 6))
        self.explorer_canvas = FigureCanvasTkAgg(self.explorer_fig, master=viz_frame)
        self.explorer_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar for interactive figure
        toolbar = NavigationToolbar2Tk(self.explorer_canvas, viz_frame)
        toolbar.update()

        # Initialize state variables
        self.theta = tk.DoubleVar(value=0.0)
        self.alpha = tk.DoubleVar(value=0.0)
        self.theta_dot = tk.DoubleVar(value=0.0)
        self.alpha_dot = tk.DoubleVar(value=0.0)
        self.voltage_change = tk.DoubleVar(value=0.0)

        # Create the sliders and controls
        self.create_explorer_controls(control_frame)

        # Initialize the plot
        self.update_explorer_visualization()

    def create_explorer_controls(self, parent):
        # Sliders for state variables
        slider_frame = ttk.LabelFrame(parent, text="State Variables", padding="10")
        slider_frame.pack(fill=tk.X, pady=10)

        # Theta slider
        ttk.Label(slider_frame, text="Theta (Arm Position):").pack(anchor=tk.W, pady=(10, 0))
        theta_slider = ttk.Scale(
            slider_frame,
            from_=-np.pi / 2,
            to=np.pi / 2,
            orient=tk.HORIZONTAL,
            variable=self.theta,
            command=lambda _: self.update_explorer_visualization(),
            length=200
        )
        theta_slider.pack(fill=tk.X, pady=(0, 10))

        # Alpha slider
        ttk.Label(slider_frame, text="Alpha (Pendulum Angle):").pack(anchor=tk.W, pady=(10, 0))
        alpha_slider = ttk.Scale(
            slider_frame,
            from_=-np.pi,
            to=np.pi,
            orient=tk.HORIZONTAL,
            variable=self.alpha,
            command=lambda _: self.update_explorer_visualization(),
            length=200
        )
        alpha_slider.pack(fill=tk.X, pady=(0, 10))

        # Theta dot slider - UPDATED RANGE
        ttk.Label(slider_frame, text="Theta Dot (Arm Velocity):").pack(anchor=tk.W, pady=(10, 0))
        theta_dot_slider = ttk.Scale(
            slider_frame,
            from_=-8.0,  # Modified from -5.0 to -8.0
            to=8.0,      # Modified from 5.0 to 8.0
            orient=tk.HORIZONTAL,
            variable=self.theta_dot,
            command=lambda _: self.update_explorer_visualization(),
            length=200
        )
        theta_dot_slider.pack(fill=tk.X, pady=(0, 10))

        # Alpha dot slider - UPDATED RANGE
        ttk.Label(slider_frame, text="Alpha Dot (Pendulum Velocity):").pack(anchor=tk.W, pady=(10, 0))
        alpha_dot_slider = ttk.Scale(
            slider_frame,
            from_=-8.0,  # Modified from -5.0 to -8.0
            to=8.0,      # Modified from 5.0 to 8.0
            orient=tk.HORIZONTAL,
            variable=self.alpha_dot,
            command=lambda _: self.update_explorer_visualization(),
            length=200
        )
        alpha_dot_slider.pack(fill=tk.X, pady=(0, 10))

        # Voltage change slider
        ttk.Label(slider_frame, text="Voltage Change:").pack(anchor=tk.W, pady=(10, 0))
        voltage_change_slider = ttk.Scale(
            slider_frame,
            from_=-5.0,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=self.voltage_change,
            command=lambda _: self.update_explorer_visualization(),
            length=200
        )
        voltage_change_slider.pack(fill=tk.X, pady=(0, 10))

        # Value display
        value_frame = ttk.LabelFrame(parent, text="Current Values", padding="10")
        value_frame.pack(fill=tk.X, pady=10)

        # Create StringVar objects for display
        self.theta_text = tk.StringVar()
        self.alpha_text = tk.StringVar()
        self.theta_dot_text = tk.StringVar()
        self.alpha_dot_text = tk.StringVar()
        self.voltage_change_text = tk.StringVar()
        self.total_reward_text = tk.StringVar()

        # Add labels to display current values
        ttk.Label(value_frame, text="Theta:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(value_frame, textvariable=self.theta_text).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(value_frame, text="Alpha:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(value_frame, textvariable=self.alpha_text).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(value_frame, text="Theta Dot:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(value_frame, textvariable=self.theta_dot_text).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(value_frame, text="Alpha Dot:").grid(row=3, column=0, sticky=tk.W)
        ttk.Label(value_frame, textvariable=self.alpha_dot_text).grid(row=3, column=1, sticky=tk.W)

        ttk.Label(value_frame, text="Voltage Change:").grid(row=4, column=0, sticky=tk.W)
        ttk.Label(value_frame, textvariable=self.voltage_change_text).grid(row=4, column=1, sticky=tk.W)

        ttk.Separator(value_frame, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=5)

        ttk.Label(value_frame, text="Total Reward:").grid(row=6, column=0, sticky=tk.W)
        ttk.Label(value_frame, textvariable=self.total_reward_text).grid(row=6, column=1, sticky=tk.W)

        # Visualization type selector
        viz_frame = ttk.LabelFrame(parent, text="Visualization Type", padding="10")
        viz_frame.pack(fill=tk.X, pady=10)

        self.viz_type = tk.StringVar(value="combined")

        ttk.Radiobutton(
            viz_frame,
            text="Reward Components",
            variable=self.viz_type,
            value="components",
            command=self.update_explorer_visualization
        ).pack(anchor=tk.W, pady=5)

        ttk.Radiobutton(
            viz_frame,
            text="System State",
            variable=self.viz_type,
            value="state",
            command=self.update_explorer_visualization
        ).pack(anchor=tk.W, pady=5)

        ttk.Radiobutton(
            viz_frame,
            text="Combined View",
            variable=self.viz_type,
            value="combined",
            command=self.update_explorer_visualization
        ).pack(anchor=tk.W, pady=5)

        # Reset button
        ttk.Button(parent, text="Reset All Values", command=self.reset_values).pack(pady=10, fill=tk.X)

    def create_reward_vs_alpha_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Reward vs Alpha")

        # Create matplotlib figure
        self.alpha_fig = Figure(figsize=(10, 8))
        self.alpha_canvas = FigureCanvasTkAgg(self.alpha_fig, master=tab)
        self.alpha_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(self.alpha_canvas, tab)
        toolbar.update()

        # Initial plot
        self.plot_reward_vs_alpha()

    def create_heatmap_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Reward Heatmap")

        # Create matplotlib figure
        self.heatmap_fig = Figure(figsize=(10, 8))
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=tab)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(self.heatmap_canvas, tab)
        toolbar.update()

        # Initial plot
        self.plot_reward_heatmap()

    def create_surface_plot_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="3D Surface")

        # Create matplotlib figure
        self.surface_fig = Figure(figsize=(10, 8))
        self.surface_canvas = FigureCanvasTkAgg(self.surface_fig, master=tab)
        self.surface_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(self.surface_canvas, tab)
        toolbar.update()

        # Initial plot
        self.plot_reward_surface()

    def create_component_comparison_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Component Comparison")

        # Create matplotlib figure
        self.component_fig = Figure(figsize=(12, 10))
        self.component_canvas = FigureCanvasTkAgg(self.component_fig, master=tab)
        self.component_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(self.component_canvas, tab)
        toolbar.update()

        # Initial plot
        self.plot_component_comparison()

    def create_swingup_trajectory_tab(self):
        """Create a new tab for visualizing the swing-up trajectory"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Swing-Up Trajectory")

        # Create controls frame
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Add sliders and controls
        ttk.Label(control_frame, text="Simulation Time (s):").pack(side=tk.LEFT, padx=(10, 5))
        self.sim_time = tk.DoubleVar(value=5.0)
        sim_time_slider = ttk.Scale(
            control_frame,
            from_=1.0,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self.sim_time,
            length=200
        )
        sim_time_slider.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(control_frame, text="Initial Velocity:").pack(side=tk.LEFT, padx=(10, 5))
        self.init_velocity = tk.DoubleVar(value=3.0)
        velocity_slider = ttk.Scale(
            control_frame,
            from_=0.1,
            to=8.0,
            orient=tk.HORIZONTAL,
            variable=self.init_velocity,
            length=200
        )
        velocity_slider.pack(side=tk.LEFT, padx=(0, 20))

        # Run simulation button
        ttk.Button(
            control_frame,
            text="Run Simulation",
            command=self.plot_swingup_trajectory
        ).pack(side=tk.LEFT, padx=10)

        # Create matplotlib figure
        self.swingup_fig = Figure(figsize=(12, 10))
        self.swingup_canvas = FigureCanvasTkAgg(self.swingup_fig, master=tab)
        self.swingup_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(self.swingup_canvas, tab)
        toolbar.update()

        # Initial plot
        self.plot_swingup_trajectory()

    def reset_values(self):
        self.theta.set(0.0)
        self.alpha.set(0.0)
        self.theta_dot.set(0.0)
        self.alpha_dot.set(0.0)
        self.voltage_change.set(0.0)
        self.update_explorer_visualization()

    def update_explorer_visualization(self):
        # Update text displays
        self.theta_text.set(f"{self.theta.get():.2f}")
        self.alpha_text.set(f"{self.alpha.get():.2f}")
        self.theta_dot_text.set(f"{self.theta_dot.get():.2f}")
        self.alpha_dot_text.set(f"{self.alpha_dot.get():.2f}")
        self.voltage_change_text.set(f"{self.voltage_change.get():.2f}")

        # Get current state
        state = [self.theta.get(), self.alpha.get(), self.theta_dot.get(), self.alpha_dot.get()]
        voltage_change = self.voltage_change.get()

        # Calculate rewards
        total_reward, components = compute_reward(state, voltage_change, self.params)
        self.total_reward_text.set(f"{total_reward:.2f}")

        # Clear figure
        self.explorer_fig.clear()

        # Select visualization based on radiobutton
        viz_type = self.viz_type.get()

        if viz_type == "components":
            self.plot_components_explorer(components, total_reward)
        elif viz_type == "state":
            self.plot_state_explorer(state, total_reward)
        else:  # combined
            self.plot_combined_explorer(state, components, total_reward)

        # Update the canvas
        self.explorer_fig.tight_layout()
        self.explorer_canvas.draw()

    def plot_components_explorer(self, components, total_reward):
        ax = self.explorer_fig.add_subplot(111)

        # Add total reward to components
        all_components = components.copy()
        all_components["Total Reward"] = total_reward

        # Extract names and values
        names = list(all_components.keys())
        values = list(all_components.values())

        # Define colors
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#000000']

        # Create horizontal bar chart
        bars = ax.barh(names, values, color=colors[:len(names)])

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width >= 0 else width - 0.3
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                    va='center', ha='left' if width >= 0 else 'right', color='black')

        ax.set_title('Reward Components')
        ax.set_xlabel('Value')
        ax.grid(axis='x', alpha=0.3)

    def plot_state_explorer(self, state, total_reward):
        ax = self.explorer_fig.add_subplot(111)

        # Extract state values
        theta, alpha, theta_dot, alpha_dot = state

        # System state visualization parameters
        cart_width = 0.4
        cart_height = 0.2
        pendulum_length = 0.3

        # Draw the cart
        cart_x = theta  # Cart position is theta
        cart = plt.Rectangle((cart_x - cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True,
                             color='blue')
        ax.add_patch(cart)

        # Draw the pendulum - UPDATED FOR NEW CONVENTION
        # For the convention where α = 0 is up and α = π is down:
        pendulum_x = cart_x
        pendulum_y = 0
        # Use negative sin for x and negative cos for y to get correct orientation (upright at 0)
        pendulum_end_x = pendulum_x + pendulum_length * np.sin(alpha)
        pendulum_end_y = pendulum_y - pendulum_length * np.cos(alpha)
        ax.plot([pendulum_x, pendulum_end_x], [pendulum_y, pendulum_end_y], 'k-', linewidth=3)

        # Draw a circle at the pendulum end
        ax.plot(pendulum_end_x, pendulum_end_y, 'ro', markersize=10)

        # Draw velocity vectors
        if theta_dot != 0:
            ax.arrow(cart_x, pendulum_y - 0.05, theta_dot / 10, 0, head_width=0.03, head_length=0.03, fc='green',
                     ec='green')

        if alpha_dot != 0:
            ax.arrow(pendulum_end_x, pendulum_end_y, alpha_dot / 10, 0, head_width=0.03, head_length=0.03, fc='purple',
                     ec='purple')

        # Set axis limits and labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title('System State')
        ax.set_xlabel('Position')
        ax.set_ylabel('Height')
        ax.grid(True)
        ax.set_aspect('equal')

        # Add reward information
        ax.text(0.05, 0.95, f'Total Reward: {total_reward:.2f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Add angle convention explanation
        ax.text(0.05, 0.05, f'α = 0: up, α = π: down', transform=ax.transAxes,
                fontsize=8, color='gray', verticalalignment='bottom')

    def plot_combined_explorer(self, state, components, total_reward):
        # Create subplots
        gs = self.explorer_fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = self.explorer_fig.add_subplot(gs[0, 0])
        ax2 = self.explorer_fig.add_subplot(gs[0, 1])

        # Plot components on the left
        all_components = components.copy()
        all_components["Total Reward"] = total_reward
        names = list(all_components.keys())
        values = list(all_components.values())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#000000']
        bars = ax1.barh(names, values, color=colors[:len(names)])

        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width >= 0 else width - 0.3
            ax1.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                     va='center', ha='left' if width >= 0 else 'right', color='black')

        ax1.set_title('Reward Components')
        ax1.set_xlabel('Value')
        ax1.grid(axis='x', alpha=0.3)

        # Plot state on the right - UPDATED ANGLE CONVENTION
        theta, alpha, theta_dot, alpha_dot = state
        cart_width = 0.4
        cart_height = 0.2
        pendulum_length = 0.3

        cart_x = theta
        cart = plt.Rectangle((cart_x - cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True,
                             color='blue')
        ax2.add_patch(cart)

        # Draw pendulum with updated angle convention
        pendulum_x = cart_x
        pendulum_y = 0
        pendulum_end_x = pendulum_x + pendulum_length * np.sin(alpha)
        pendulum_end_y = pendulum_y - pendulum_length * np.cos(alpha)
        ax2.plot([pendulum_x, pendulum_end_x], [pendulum_y, pendulum_end_y], 'k-', linewidth=3)
        ax2.plot(pendulum_end_x, pendulum_end_y, 'ro', markersize=10)

        if theta_dot != 0:
            ax2.arrow(cart_x, pendulum_y - 0.05, theta_dot / 10, 0, head_width=0.03, head_length=0.03, fc='green',
                      ec='green')

        if alpha_dot != 0:
            ax2.arrow(pendulum_end_x, pendulum_end_y, alpha_dot / 10, 0, head_width=0.03, head_length=0.03, fc='purple',
                      ec='purple')

        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_title('System State')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Height')
        ax2.grid(True)
        ax2.set_aspect('equal')

        ax2.text(0.05, 0.95, f'Total Reward: {total_reward:.2f}', transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Add angle convention explanation
        ax2.text(0.05, 0.05, f'α = 0: up, α = π: down', transform=ax2.transAxes,
                 fontsize=8, color='gray', verticalalignment='bottom')

    def plot_reward_vs_alpha(self):
        self.status_var.set("Generating Reward vs Alpha plot...")
        self.root.update()

        # Clear the figure
        self.alpha_fig.clear()
        ax = self.alpha_fig.add_subplot(111)

        # Create a range of alpha values
        alpha_values = np.linspace(-np.pi, np.pi, 1000)

        # Initialize arrays for components
        upright_rewards = []
        velocity_penalties = []
        pos_penalties = []
        bonuses = []
        limit_penalties = []
        energy_rewards = []
        voltage_penalties = []
        total_rewards = []

        # Set other state variables as constants for this plot
        theta = 0.0
        theta_dot = 0.0
        alpha_dot = 0.0
        voltage_change = 0.0

        # Calculate rewards for each alpha value
        for alpha in alpha_values:
            state = [theta, alpha, theta_dot, alpha_dot]
            total, components = compute_reward(state, voltage_change, self.params)

            upright_rewards.append(components["Upright Reward"])
            velocity_penalties.append(components["Velocity Penalty"])
            pos_penalties.append(components["Position Penalty"])
            bonuses.append(components["Upright Bonus"])
            limit_penalties.append(components["Limit Penalty"])
            energy_rewards.append(components["Energy Reward"])
            voltage_penalties.append(components["Voltage Change"])
            total_rewards.append(total)

        # Create the plot
        ax.plot(alpha_values, upright_rewards, label='Upright Reward')
        ax.plot(alpha_values, velocity_penalties, label='Velocity Penalty')
        ax.plot(alpha_values, pos_penalties, label='Position Penalty')
        ax.plot(alpha_values, bonuses, label='Upright Bonus')
        ax.plot(alpha_values, limit_penalties, label='Limit Penalty')
        ax.plot(alpha_values, energy_rewards, label='Energy Reward')
        ax.plot(alpha_values, voltage_penalties, label='Voltage Change')
        ax.plot(alpha_values, total_rewards, 'k--', linewidth=2, label='Total Reward')

        # Annotate key positions
        ax.axvline(x=0, color='g', linestyle='--', alpha=0.3, label='Up Position (α=0)')
        ax.axvline(x=np.pi, color='r', linestyle='--', alpha=0.3, label='Down Position (α=π)')
        ax.axvline(x=-np.pi, color='r', linestyle='--', alpha=0.3, label='Down Position (α=-π)')

        ax.set_title('Reward Components vs Pendulum Angle (alpha)')
        ax.set_xlabel('Pendulum Angle (alpha)')
        ax.set_ylabel('Reward Value')
        ax.legend()
        ax.grid(True)

        # Update the figure
        self.alpha_fig.tight_layout()
        self.alpha_canvas.draw()

        self.status_var.set("Reward vs Alpha plot ready")

    def plot_reward_heatmap(self):
        self.status_var.set("Generating Reward Heatmap plot...")
        self.root.update()

        # Clear the figure
        self.heatmap_fig.clear()
        ax = self.heatmap_fig.add_subplot(111)

        # Create meshgrid for alpha and alpha_dot - UPDATED RANGE
        alpha_values = np.linspace(-np.pi, np.pi, 100)
        alpha_dot_values = np.linspace(-8, 8, 100)  # Modified from -5, 5 to -8, 8
        alpha_grid, alpha_dot_grid = np.meshgrid(alpha_values, alpha_dot_values)

        # Fixed values for other variables
        theta = 0.0
        theta_dot = 0.0
        voltage_change = 0.0

        # Calculate reward for each point in the grid
        rewards = np.zeros(alpha_grid.shape)
        for i in range(len(alpha_values)):
            for j in range(len(alpha_dot_values)):
                state = [theta, alpha_grid[j, i], theta_dot, alpha_dot_grid[j, i]]
                rewards[j, i], _ = compute_reward(state, voltage_change, self.params)

        # Create a custom colormap
        colors = [(0.8, 0, 0), (1, 1, 1), (0, 0.8, 0)]  # Red -> White -> Green
        cmap = LinearSegmentedColormap.from_list('RWG', colors, N=256)

        # Create the plot
        im = ax.contourf(alpha_grid, alpha_dot_grid, rewards, 50, cmap=cmap)
        self.heatmap_fig.colorbar(im, ax=ax, label='Reward')
        ax.set_title('Reward Heatmap: Pendulum Angle vs Angular Velocity')
        ax.set_xlabel('Pendulum Angle (alpha)')
        ax.set_ylabel('Pendulum Angular Velocity (alpha_dot)')

        # Add annotations for key positions
        ax.axvline(x=0, color='w', linestyle='--', alpha=0.5, label='Up Position (α=0)')
        ax.axvline(x=np.pi, color='k', linestyle='--', alpha=0.3, label='Down Position (α=π)')
        ax.axvline(x=-np.pi, color='k', linestyle='--', alpha=0.3, label='Down Position (α=-π)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        # Update the figure
        self.heatmap_fig.tight_layout()
        self.heatmap_canvas.draw()

        self.status_var.set("Reward Heatmap plot ready")

    def plot_reward_surface(self):
        self.status_var.set("Generating 3D Surface plot...")
        self.root.update()

        # Clear the figure
        self.surface_fig.clear()
        ax = self.surface_fig.add_subplot(111, projection='3d')

        # Create meshgrid - UPDATED RANGE
        alpha_values = np.linspace(-np.pi, np.pi, 50)
        alpha_dot_values = np.linspace(-8, 8, 50)  # Modified from -5, 5 to -8, 8
        alpha_grid, alpha_dot_grid = np.meshgrid(alpha_values, alpha_dot_values)

        # Fixed values
        theta = 0.0
        theta_dot = 0.0
        voltage_change = 0.0

        # Calculate reward for each point
        rewards = np.zeros(alpha_grid.shape)
        for i in range(len(alpha_values)):
            for j in range(len(alpha_dot_values)):
                state = [theta, alpha_grid[j, i], theta_dot, alpha_dot_grid[j, i]]
                rewards[j, i], _ = compute_reward(state, voltage_change, self.params)

        # Plot surface
        surf = ax.plot_surface(alpha_grid, alpha_dot_grid, rewards, cmap='viridis',
                               linewidth=0, antialiased=True, alpha=0.8)

        # Add colorbar
        self.surface_fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Reward')

        # Set labels and title
        ax.set_xlabel('Pendulum Angle (alpha)')
        ax.set_ylabel('Angular Velocity (alpha_dot)')
        ax.set_zlabel('Reward')
        ax.set_title('3D Reward Surface')

        # Add text annotations for key positions - in 3D space
        ax.text(-np.pi, 0, np.min(rewards), "Down Position (α=-π)", color='red')
        ax.text(0, 0, np.min(rewards), "Up Position (α=0)", color='green')
        ax.text(np.pi, 0, np.min(rewards), "Down Position (α=π)", color='red')

        # Update the figure
        self.surface_fig.tight_layout()
        self.surface_canvas.draw()

        self.status_var.set("3D Surface plot ready")

    def plot_component_comparison(self):
        self.status_var.set("Generating Component Comparison plots...")
        self.root.update()

        # Clear the figure
        self.component_fig.clear()

        # Create a grid of subplots
        axs = self.component_fig.subplots(2, 4)
        axs = axs.flatten()

        # Create meshgrid for alpha and alpha_dot - UPDATED RANGE
        alpha_values = np.linspace(-np.pi, np.pi, 50)
        alpha_dot_values = np.linspace(-8, 8, 50)  # Modified from -5, 5 to -8, 8
        alpha_grid, alpha_dot_grid = np.meshgrid(alpha_values, alpha_dot_values)

        # Create state grid with fixed values for theta and theta_dot
        theta = np.zeros_like(alpha_grid)
        theta_dot = np.zeros_like(alpha_grid)
        state_grid = [theta, alpha_grid, theta_dot, alpha_dot_grid]

        # Component names and titles
        components = [
            "Upright Reward",
            "Velocity Penalty",
            "Position Penalty",
            "Upright Bonus",
            "Limit Penalty",
            "Energy Reward",
            "Voltage Change"
        ]

        # Compute and plot each component
        for i, component in enumerate(components):
            rewards = compute_component(component, state_grid, 0.0, self.params)
            im = axs[i].contourf(alpha_grid, alpha_dot_grid, rewards, 50, cmap='coolwarm')
            axs[i].set_title(component)
            axs[i].set_xlabel('Alpha')
            axs[i].set_ylabel('Alpha Dot')
            axs[i].axvline(x=0, color='w', linestyle='--', alpha=0.3, label='Up')
            axs[i].axvline(x=np.pi, color='k', linestyle='--', alpha=0.3, label='Down')
            axs[i].axvline(x=-np.pi, color='k', linestyle='--', alpha=0.3)
            axs[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            self.component_fig.colorbar(im, ax=axs[i])

        # Calculate total reward for the last subplot
        total_rewards = np.zeros(alpha_grid.shape)
        for i in range(alpha_grid.shape[0]):
            for j in range(alpha_grid.shape[1]):
                state = [theta[i, j], alpha_grid[i, j], theta_dot[i, j], alpha_dot_grid[i, j]]
                total_rewards[i, j], _ = compute_reward(state, 0.0, self.params)

        # Plot total reward
        im = axs[7].contourf(alpha_grid, alpha_dot_grid, total_rewards, 50, cmap='coolwarm')
        axs[7].set_title('Total Reward')
        axs[7].set_xlabel('Alpha')
        axs[7].set_ylabel('Alpha Dot')
        axs[7].axvline(x=0, color='w', linestyle='--', alpha=0.3, label='Up')
        axs[7].axvline(x=np.pi, color='k', linestyle='--', alpha=0.3, label='Down')
        axs[7].axvline(x=-np.pi, color='k', linestyle='--', alpha=0.3)
        axs[7].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.component_fig.colorbar(im, ax=axs[7])

        # Update the figure
        self.component_fig.tight_layout()
        self.component_canvas.draw()

        self.status_var.set("Component Comparison plots ready")

    def simulate_pendulum_trajectory(self):
        """Simulate a trajectory of the pendulum swinging from down to up position
           with updated angle convention"""
        # Get simulation time and initial velocity from sliders
        sim_time = self.sim_time.get()
        init_velocity = self.init_velocity.get()
        dt = 0.01

        # Time points
        time_points = np.arange(0, sim_time, dt)

        # Create a trajectory that swings up from bottom (π) to top (0)
        # Using a cosine function to create a smooth swing-up
        # In new convention, π is down and 0 is up
        alphas = np.pi * np.cos(np.pi * time_points / sim_time)

        # Calculate velocity based on the derivative of position
        # Scale by the initial velocity parameter
        alpha_dots = -np.pi ** 2 / sim_time * np.sin(np.pi * time_points / sim_time)
        alpha_dots = alpha_dots * (init_velocity / np.max(np.abs(alpha_dots)))

        # Arm position and velocity remain at zero
        thetas = np.zeros_like(time_points)
        theta_dots = np.zeros_like(time_points)

        # Calculate rewards along the trajectory
        rewards = []
        components_dict = {
            "Upright Reward": [],
            "Velocity Penalty": [],
            "Position Penalty": [],
            "Upright Bonus": [],
            "Limit Penalty": [],
            "Energy Reward": [],
            "Voltage Change": []
        }

        for i in range(len(time_points)):
            state = [thetas[i], alphas[i], theta_dots[i], alpha_dots[i]]
            reward, components = compute_reward(state, 0.0, self.params)
            rewards.append(reward)

            # Store each component
            for key in components_dict:
                components_dict[key].append(components[key])

        return {
            'time': time_points,
            'alpha': alphas,
            'alpha_dot': alpha_dots,
            'theta': thetas,
            'theta_dot': theta_dots,
            'reward': np.array(rewards),
            'components': components_dict
        }

    def plot_swingup_trajectory(self):
        """Plot the reward and pendulum state during swing-up trajectory"""
        self.status_var.set("Simulating pendulum swing-up trajectory...")
        self.root.update()

        # Run simulation
        trajectory = self.simulate_pendulum_trajectory()

        # Clear figure
        self.swingup_fig.clear()

        # Create 3 subplots: rewards over time, components over time, and pendulum states
        gs = self.swingup_fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
        ax_reward = self.swingup_fig.add_subplot(gs[0])
        ax_components = self.swingup_fig.add_subplot(gs[1])
        ax_state = self.swingup_fig.add_subplot(gs[2])

        # Plot total reward over time
        ax_reward.plot(trajectory['time'], trajectory['reward'], 'b-', linewidth=2)
        ax_reward.set_title('Total Reward during Swing-Up')
        ax_reward.set_ylabel('Reward')
        ax_reward.grid(True)

        # Plot reward components over time
        components = trajectory['components']
        ax_components.plot(trajectory['time'], components['Upright Reward'], label='Upright Reward')
        ax_components.plot(trajectory['time'], components['Upright Bonus'], label='Upright Bonus')
        ax_components.plot(trajectory['time'], components['Energy Reward'], label='Energy Reward')
        ax_components.plot(trajectory['time'], components['Velocity Penalty'], label='Velocity')
        ax_components.set_title('Reward Components during Swing-Up')
        ax_components.set_ylabel('Component Value')
        ax_components.legend(loc='upper right', fontsize='small')
        ax_components.grid(True)

        # Create pendulum visualization at key points
        num_points = 5
        indices = np.linspace(0, len(trajectory['time']) - 1, num_points, dtype=int)

        # Set up the axes for pendulum visualization
        ax_state.set_xlim(-1.5, 5.5)
        ax_state.set_ylim(-1.5, 1.5)
        ax_state.set_aspect('equal')
        ax_state.grid(True)
        ax_state.set_title('Pendulum States during Swing-Up (α = 1: up, α = -1: down)')
        ax_state.set_xlabel('Time (s)')

        # Pendulum visualization parameters
        pendulum_length = 1.0
        colors = plt.cm.viridis(np.linspace(0, 1, num_points))

        # Draw pendulums at different time points with updated angle convention
        for i, idx in enumerate(indices):
            # Get state at this point
            alpha = trajectory['alpha'][idx] - np.pi  # Convert to new convention
            alpha_dot = trajectory['alpha_dot'][idx]
            time = trajectory['time'][idx]
            reward = trajectory['reward'][idx]

            # Calculate pendulum position with updated angle convention
            # For convention where 0 is up and π is down:
            pendulum_x = 0
            pendulum_y = 0
            pendulum_end_x = pendulum_x + pendulum_length * np.sin(alpha)
            pendulum_end_y = pendulum_y - pendulum_length * np.cos(alpha)

            # Plot pendulum
            ax_state.plot([pendulum_x, pendulum_end_x], [pendulum_y, pendulum_end_y],
                          color=colors[i], linewidth=2,
                          label=f't={time:.2f}s, α={alpha:.2f}, R={reward:.2f}')

            # Plot pendulum end as circle
            ax_state.plot(pendulum_end_x, pendulum_end_y, 'o', color=colors[i], markersize=8)

            # Show velocity vector if significant
            if abs(alpha_dot) > 0.1:
                # Scale velocity for visibility
                vel_scale = 0.2
                # Show velocity perpendicular to pendulum arm
                # Calculate the perpendicular direction for angular velocity
                # For the new convention, α = 0 is up
                perp_x = np.cos(alpha)
                perp_y = np.sin(alpha)

                ax_state.arrow(
                    pendulum_end_x, pendulum_end_y,
                    vel_scale * alpha_dot * perp_x,
                    vel_scale * alpha_dot * perp_y,
                    head_width=0.05, head_length=0.1, fc=colors[i], ec=colors[i]
                )

        # Add legend to the state plot
        ax_state.legend(loc='upper right', fontsize='small')

        # Update the figure
        self.swingup_fig.tight_layout()
        self.swingup_canvas.draw()

        self.status_var.set("Swing-up trajectory simulation complete")


# Main application function
def main():
    root = tk.Tk()
    app = RewardExplorerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()