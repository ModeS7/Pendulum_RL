from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from torchrl.envs import EnvBase, TransformedEnv
from tensordict import TensorDict
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from torch.nn.functional import dropout

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

# Constants
DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x):
    """Normalize angle to [-pi, pi]"""
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


def _step(tensordict):
    """
    Perform one step of the pendulum simulation using RK4 integration.
    """
    # Extract state variables
    th, thdot = tensordict["th"], tensordict["thdot"]
    phi, phidot = tensordict["phi"], tensordict["phidot"]

    # Extract parameters
    g = tensordict["params", "g"]  # Gravity constant (m/s²)
    Mp = tensordict["params", "m"]  # Pendulum mass (kg)
    Lp = tensordict["params", "l"]  # Pendulum length (m)
    dt = tensordict["params", "dt"]

    # Extract action
    u = tensordict["action"].squeeze(-1)

    # Debugging
    # debug_mode = False  # Set to True when you want to see debugging info
    # if debug_mode and th.ndim > 0 and th.shape[0] > 0:
    #     print(f"DEBUG - Input to _step: th={th[0].item():.4f}, thdot={thdot[0].item():.4f}, "
    #           f"phi={phi[0].item():.4f}, phidot={phidot[0].item():.4f}, action={u[0].item():.4f}")


    # Constants
    Rm = 8.4  # Motor resistance (Ohm)
    kt = 0.042  # Motor torque constant (N·m/A)
    km = 0.042  # Motor back-EMF constant (V·s/rad)
    Jm = 4e-6  # Motor moment of inertia (kg·m²)
    Jh = 0.6e-6  # Hub moment of inertia (kg·m^2)
    Mr = 0.095  # Rotary arm mass (kg)
    Lr = 0.085  # Arm length, pivot to end (m)
    Jp = (1 / 3) * Mp * Lp ** 2  # Pendulum moment of inertia (kg·m²)
    Br = 0.001  # Rotary arm viscous damping coefficient (N·m·s/rad)
    Bp = 0.0001  # Pendulum viscous damping coefficient (N·m·s/rad)
    Jr = Jm + Jh + Mr * Lr ** 2 / 3  # Arm inertia

    # Define derivative function for RK4 integration
    def f(th, thdot, phi, phidot):
        # Motor current and torque
        im = (u - km * phidot) / Rm
        tau = kt * im

        # Inertia matrix elements
        M11 = Jr + Mp * Lr ** 2
        M12 = Mp * Lr * (Lp / 2) * torch.cos(th)
        M21 = M12
        M22 = Jp
        det_M = M11 * M22 - M12 * M21

        # Coriolis and gravitational terms
        C1 = -Mp * Lr * (Lp / 2) * thdot ** 2 * torch.sin(th) - Br * phidot
        C2 = Mp * g * (Lp / 2) * torch.sin(th) - Bp * thdot

        # Solve for accelerations
        phidotdot = (M22 * (tau + C1) - M12 * (0 + C2)) / det_M
        thdotdot = (M11 * (0 + C2) - M21 * (tau + C1)) / det_M

        return thdot, thdotdot, phidot, phidotdot

    # RK4 increments
    k1 = f(th, thdot, phi, phidot)
    k2 = f(th + (dt / 2) * k1[0], thdot + (dt / 2) * k1[1], phi + (dt / 2) * k1[2], phidot + (dt / 2) * k1[3])
    k3 = f(th + (dt / 2) * k2[0], thdot + (dt / 2) * k2[1], phi + (dt / 2) * k2[2], phidot + (dt / 2) * k2[3])
    k4 = f(th + dt * k3[0], thdot + dt * k3[1], phi + dt * k3[2], phidot + dt * k3[3])

    # Update state with RK4 weighted average
    new_th = th + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    new_thdot = thdot + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    new_phi = phi + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    new_phidot = phidot + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

    # Compute costs/rewards
    costs = (angle_normalize(th) ** 2.0 + 0.1 * thdot ** 2.0 + 0.001 * (u ** 2.0)) * 0.1
    costs = torch.clamp(costs, 0.0, 10.0)
    reward = -costs.view(*tensordict.shape, 1)

    # Create output TensorDict
    done = torch.zeros_like(reward, dtype=torch.bool)
    out = TensorDict(
        {
            "th": new_th,
            "thdot": new_thdot,
            "phi": new_phi,
            "phidot": new_phidot,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    return out


def _reset(self, tensordict):
    """Reset the environment to a random state."""
    if tensordict is None or tensordict.is_empty():
        tensordict = self.gen_params(batch_size=self.batch_size)

    # Define state bounds
    high_th = torch.tensor(DEFAULT_X, device=self.device)
    high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
    low_th = -high_th
    low_thdot = -high_thdot

    # Generate random initial state
    th = torch.rand(tensordict.shape, generator=self.rng, device=self.device) * (high_th - low_th) + low_th
    thdot = torch.rand(tensordict.shape, generator=self.rng, device=self.device) * (high_thdot - low_thdot) + low_thdot

    # Create output TensorDict
    out = TensorDict(
        {
            "th": th,
            "thdot": thdot,
            "phi": torch.zeros_like(th),
            "phidot": torch.zeros_like(th),
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
        device=self.device,
    )
    return out


def _make_spec(self, td_params):
    """Create observation, state, and action specs for the environment."""
    # Create observation spec
    self.observation_spec = CompositeSpec(
        th=UnboundedContinuousTensorSpec(shape=(), dtype=torch.float32),
        thdot=UnboundedContinuousTensorSpec(shape=(), dtype=torch.float32),
        phi=UnboundedContinuousTensorSpec(shape=(), dtype=torch.float32),
        phidot=UnboundedContinuousTensorSpec(shape=(), dtype=torch.float32),
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )

    # Set state spec
    self.state_spec = self.observation_spec.clone()

    # Set action spec
    self.action_spec = UnboundedContinuousTensorSpec(shape=(), dtype=torch.float32)

    # Set reward spec
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    """Convert a TensorDict to a CompositeSpec structure."""
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


def _set_seed(self, seed: Optional[int]):
    """Set random seed for reproducibility."""
    # Set global seed
    torch.manual_seed(seed)

    # Create device-specific generator
    if self.device.type == 'cuda':
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
    else:
        self.rng = torch.manual_seed(seed)


def gen_params(g=9.81, batch_size=None, device=None) -> TensorDictBase:
    """Generate environment parameters."""
    if batch_size is None:
        batch_size = []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create parameter TensorDict
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "dt": torch.tensor(0.05, device=device),
                    "g": torch.tensor(g, device=device),
                    "m": torch.tensor(0.024, device=device),
                    "l": torch.tensor(0.129 / 2.0, device=device),
                },
                [],
                device=device,
            )
        },
        [],
        device=device,
    )

    # Expand to batch size if needed
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


class PendulumEnv(EnvBase):
    """
    Pendulum environment simulation.

    Simulates a rotary pendulum (Qube Servo 2) with motor dynamics.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """Initialize pendulum environment."""
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Define methods
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec
    _reset = _reset

    def _step(self, tensordict):
        """Step the environment."""
        # The issue is that we're not properly using the result of the base environment step
        # in the transformed environment's step method

        # First, create a deep clone of the input tensordict to avoid modifying the original
        td_clone = tensordict.clone()

        # Apply the physics step function to get the next state
        result = _step(td_clone)

        # Debug output to verify state changes (uncomment if needed)
        # print(f"ENV STEP: Input th={tensordict['th'].item():.4f}, Output th={result['th'].item():.4f}")

        return result
    _set_seed = _set_seed


class SinTransform(Transform):
    """Transform that calculates the sine of input values."""

    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.sin()

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1, high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


class CosTransform(Transform):
    """Transform that calculates the cosine of input values."""

    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.cos()

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1, high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


class ExplorationPolicy(TensorDictModule):
    """Policy wrapper that adds exploration noise to actions."""

    def __init__(self, policy, noise_scale=0.5):
        dummy_module = nn.Identity()
        super().__init__(
            module=dummy_module,
            in_keys=policy.in_keys,
            out_keys=policy.out_keys
        )
        self.policy = policy
        self.noise_scale = noise_scale

    def forward(self, tensordict):
        # Get base action from policy
        action_td = self.policy(tensordict)

        # Add exploration noise
        noise = torch.randn_like(action_td["action"]) * self.noise_scale
        action_td["action"] = action_td["action"] + noise

        return action_td


def setup_environment():
    """
    Set up the pendulum environment with all necessary transforms.

    Returns:
        TransformedEnv: The environment ready for training
    """
    # Create base environment
    env = PendulumEnv()

    # Add unsqueeze transform
    env = TransformedEnv(
        env,
        UnsqueezeTransform(
            dim=-1,
            in_keys=["th", "thdot", "phi", "phidot"],
            in_keys_inv=["th", "thdot", "phi", "phidot"],
        ),
    )

    # Add sin/cos transforms for angles
    t_sin = SinTransform(in_keys=["th"], out_keys=["sin_th"])
    t_cos = CosTransform(in_keys=["th"], out_keys=["cos_th"])
    p_sin = SinTransform(in_keys=["phi"], out_keys=["sin_phi"])
    p_cos = CosTransform(in_keys=["phi"], out_keys=["cos_phi"])

    env.append_transform(t_sin)
    env.append_transform(t_cos)
    env.append_transform(p_sin)
    env.append_transform(p_cos)

    # Add concatenation transform
    cat_transform = CatTensors(
        in_keys=["sin_th", "cos_th", "thdot", "sin_phi", "cos_phi", "phidot"],
        dim=-1,
        out_key="observation",
        del_keys=False
    )
    env.append_transform(cat_transform)

    # Verify environment specs
    check_env_specs(env)

    return env


def create_policy_network(env):
    """
    Create a policy network for the pendulum environment.

    Args:
        env: The environment to create the policy for

    Returns:
        policy: The policy network as a TensorDictModule
    """
    device = env.device

    # Define network architecture
    net = nn.Sequential(
        nn.Linear(6, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 1),
    ).to(device)

    # Initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)

    # Create policy module
    policy = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["action"],
    )

    return policy


def debug_policy(policy, env, steps=10):
    """Debug policy behavior with the full transformed environment."""
    # Reset environment
    td = env.reset()

    # Helper function to safely get item values from tensors
    def safe_item(tensor):
        if tensor.ndim > 0 and tensor.size(0) > 1:
            return tensor[0].item()
        return tensor.item()

    print("Initial state:",
          f"th={safe_item(td['th']):.4f}",
          f"thdot={safe_item(td['thdot']):.4f}",
          f"phi={safe_item(td['phi']):.4f}",
          f"phidot={safe_item(td['phidot']):.4f}")

    print("\nPolicy outputs and state transitions:")
    for i in range(steps):
        # Store original state for display
        orig_th = safe_item(td['th'])
        orig_thdot = safe_item(td['thdot'])
        orig_phi = safe_item(td['phi'])
        orig_phidot = safe_item(td['phidot'])

        # Get action from policy
        with torch.no_grad():
            action_td = policy(td.clone())

        # Get scalar action value for display
        action = safe_item(action_td["action"])

        print(f"Step {i}: State=[{orig_th:.4f}, {orig_thdot:.4f}, "
              f"{orig_phi:.4f}, {orig_phidot:.4f}], Action={action:.4f}")

        # Create a new tensordict with the action
        td_with_action = td.clone()
        td_with_action["action"] = action_td["action"]

        # Step environment and explicitly capture the returned tensordict
        td_next = env.step(td_with_action)

        # CRITICAL: Assign the new state to td for the next iteration
        # This is the key line that was likely causing your issue
        td = td_next

        # Print resulting state
        print(f"  → Next state: [{safe_item(td['th']):.4f}, {safe_item(td['thdot']):.4f}, "
              f"{safe_item(td['phi']):.4f}, {safe_item(td['phidot']):.4f}]")



def test_environment_step(env):
    """Test if the environment step function is working correctly."""
    # Get base environment
    base_env = env
    while hasattr(base_env, "base_env"):
        base_env = base_env.base_env

    # Reset environment
    td = base_env.reset()

    print(f"Initial state: th={td['th'].item():.4f}, thdot={td['thdot'].item():.4f}")

    # Apply a constant action
    action_value = 1.0
    td["action"] = torch.tensor([action_value], device=td.device)

    # Step directly with the base environment's _step method
    new_td = base_env._step(td)

    print(f"After _step: th={new_td['th'].item():.4f}, thdot={new_td['thdot'].item():.4f}")

    # Step using the env.step method
    td["action"] = torch.tensor([action_value], device=td.device)
    step_td = env.step(td)

    print(f"After env.step: th={step_td['th'].item():.4f}, thdot={step_td['thdot'].item():.4f}")

    # Check if the state is being updated
    if td['th'].item() != new_td['th'].item():
        print("_step is correctly updating the state")
    else:
        print("_step is NOT updating the state")

    if td['th'].item() != step_td['th'].item():
        print("env.step is correctly updating the state")
    else:
        print("env.step is NOT updating the state")


def train_policy(env, policy, iterations=2000, batch_size=32):
    """
    Train a policy on the pendulum environment.

    Args:
        env: The environment to train on
        policy: The policy to train
        iterations: Number of training iterations
        batch_size: Batch size for training

    Returns:
        policy: The trained policy
        logs: Training logs
    """
    # Create exploration policy
    exploration_policy = ExplorationPolicy(policy, noise_scale=1.0)

    # Create optimizer and scheduler
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, batch_size * iterations)

    # Set up training loop
    pbar = tqdm.tqdm(range(iterations))
    logs = defaultdict(list)
    best_return = float('-inf')
    patience = 0
    max_patience = 50
    noise_scale = 1.0

    for i in pbar:
        # Reduce exploration noise over time
        if i % 100 == 0 and noise_scale > 0.1:
            noise_scale *= 0.9
            exploration_policy.noise_scale = noise_scale

        # Reset environment with batch parameters
        init_td = env.reset(env.gen_params(batch_size=[batch_size]))

        # We'll use the rollout function but make sure it's handling state properly
        try:
            # Use the env's rollout function which should handle state management internally
            rollout = env.rollout(100, exploration_policy, tensordict=init_td, auto_reset=False)
        except Exception as e:
            print(f"Error in rollout: {e}")
            # If rollout fails, we could implement a manual rollout here
            # For now, we'll just continue to the next iteration
            continue

        # Compute loss and update policy
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

        optim.step()
        optim.zero_grad()

        # Track best model
        if traj_return > best_return:
            best_return = traj_return
            # Save best model
            torch.save(policy.state_dict(), "best_pendulum_policy.pt")
            patience = 0
        else:
            patience += 1

        # Early stopping
        if patience > max_patience:
            print(f"Early stopping at iteration {i}")
            # Load best model
            policy.load_state_dict(torch.load("best_pendulum_policy.pt"))
            break

        # Update progress bar
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, "
            f"noise: {noise_scale:.3f}, grad norm: {grad_norm: 4.4}"
        )

        # Log metrics
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", 'reward'].mean().item())

        # Step scheduler
        scheduler.step()

        # Occasionally debug policy
        if i % 50 == 0:  # Reduced frequency to 50 from 10
            debug_policy(policy, env)

    return policy, logs


def plot_training_results(logs):
    """
    Plot training results.

    Args:
        logs: Training logs dictionary
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(logs["return"])
    plt.title("Returns")
    plt.xlabel("Iteration")

    plt.subplot(1, 2, 2)
    plt.plot(logs["last_reward"])
    plt.title("Last Reward")
    plt.xlabel("Iteration")

    plt.tight_layout()
    plt.show()


def visualize_policy(policy, env, steps=100, save_path=None):
    """
    Visualize a policy by running it on the environment and creating an animation.

    Args:
        policy: The policy to visualize
        env: The environment to use
        steps: Number of steps to simulate
        save_path: Path to save the animation

    Returns:
        animation: The animation object
        history: History of states and actions
    """
    # Reset environment
    td = env.reset()

    # Lists to store state history
    history = {
        'theta': [],
        'phi': [],
        'action': []
    }

    # Record initial state
    history['theta'].append(td["th"].cpu().item())
    history['phi'].append(td["phi"].cpu().item())

    # Collect simulation data
    for i in range(steps):
        # Get action from policy
        with torch.no_grad():
            action_td = policy(td.clone())
            action = action_td["action"].item()

        # Record action
        history['action'].append(action)

        # Apply action to environment
        td["action"] = action_td["action"]

        # Step environment
        td = env.step(td)

        # Record state
        history['theta'].append(td["th"].item())
        history['phi'].append(td["phi"].item())

        if td["done"].any():
            break

    # Create figure for animation
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    # Main simulation plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title('Qube Servo 2 Simulation')

    # Create pendulum visualization elements
    base_width = 0.4
    base = plt.Rectangle((-base_width / 2, -base_width / 2), base_width, base_width, color='gray')
    arm_line, = ax1.plot([], [], 'r-', lw=2)  # Motor arm
    pendulum_line, = ax1.plot([], [], 'b-', lw=2)  # Pendulum
    pendulum_bob = plt.Circle((0, 0), 0.1, color='blue')

    ax1.add_patch(base)
    ax1.add_patch(pendulum_bob)

    # Plots for tracking variables
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(0, steps)
    ax2.set_ylim(-5 * np.pi, 5 * np.pi)
    ax2.set_title('Angles')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Angle (rad)')
    ax2.grid(True)

    theta_line, = ax2.plot([], [], 'b-', label='Theta')
    phi_line, = ax2.plot([], [], 'r-', label='Phi')
    ax2.legend()

    # Plot for control actions
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_xlim(0, steps)

    # Determine action range from data
    actions = np.array(history['action'])
    action_range = max(5.0, np.max(np.abs(actions))) if len(actions) > 0 else 5.0
    ax3.set_ylim(-action_range, action_range)

    ax3.set_title('Control Action')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Voltage (V)')
    ax3.grid(True)

    action_line, = ax3.plot([], [], 'g-')

    # Animation update function
    def update(i):
        # Update pendulum visualization
        theta = history['theta'][i]
        phi = history['phi'][i]

        # Motor arm position
        arm_length = 0.3
        arm_x = arm_length * np.sin(phi)
        arm_y = arm_length * np.cos(phi)
        arm_line.set_data([0, arm_x], [0, arm_y])

        # Pendulum position
        rod_length = 1.0
        bob_x = arm_x + rod_length * np.sin(theta)
        bob_y = arm_y + rod_length * np.cos(theta)
        pendulum_line.set_data([arm_x, bob_x], [arm_y, bob_y])
        pendulum_bob.set_center((bob_x, bob_y))

        # Update plots
        theta_line.set_data(range(i + 1), history['theta'][:i + 1])
        phi_line.set_data(range(i + 1), history['phi'][:i + 1])

        # Handle the action plot
        if i < len(history['action']):
            action_line.set_data(range(len(history['action'][:i + 1])), history['action'][:i + 1])
        else:
            action_line.set_data(range(len(history['action'])), history['action'])

        return arm_line, pendulum_line, pendulum_bob, theta_line, phi_line, action_line

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(history['theta']),
                        interval=100, blit=True)

    plt.tight_layout()

    # Save if requested
    if save_path:
        print(f"Saving animation to {save_path}...")
        writer = 'pillow' if save_path.endswith('.gif') else None
        try:
            ani.save(save_path, writer=writer, fps=30)
            print("Animation saved!")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            plt.savefig(save_path.replace('.gif', '.png'))
            print(f"Saved still image instead.")

    return ani, history


def evaluate_and_visualize(policy, env, save_path=None):
    """
    Evaluate and visualize a trained policy.

    Args:
        policy: The policy to evaluate
        env: The environment to use
        save_path: Path to save the animation

    Returns:
        animation: The animation object
        history: History of states and actions
    """
    # Run visualization
    print("Visualizing trained policy...")
    animation, history = visualize_policy(policy, env, steps=200, save_path=save_path)

    # Evaluate policy performance
    print("Evaluating policy performance...")
    with torch.no_grad():
        td = env.reset()
        rollout = env.rollout(100, policy, tensordict=td, auto_reset=False)

    # Calculate metrics
    avg_reward = rollout["next", "reward"].mean().item()
    total_reward = rollout["next", "reward"].sum().item()

    print(f"Average reward: {avg_reward:.6f}")
    print(f"Total reward: {total_reward:.6f}")

    return animation, history

def create_debug_env():
    """Create a basic pendulum environment without transforms for debugging"""
    return PendulumEnv()


from typing import Optional
import torch
from torchrl.envs import EnvBase, TransformedEnv
from tensordict import TensorDict


class FixedTransformedEnv(EnvBase):
    """
    Wrapper around TransformedEnv that ensures state updates are properly handled.
    """

    def __init__(self, transformed_env):
        """
        Initialize with a transformed environment

        Args:
            transformed_env: The transformed environment to wrap
        """
        super().__init__(
            device=transformed_env.device,
            batch_size=transformed_env.batch_size
        )
        self.env = transformed_env

        # Copy specs from the wrapped environment
        self.observation_spec = transformed_env.observation_spec
        self.state_spec = transformed_env.state_spec
        self.action_spec = transformed_env.action_spec
        self.reward_spec = transformed_env.reward_spec

    def _reset(self, tensordict=None, **kwargs):
        """Reset the environment."""
        return self.env.reset(tensordict, **kwargs)

    def _step(self, tensordict):
        """
        Step the environment, ensuring state is updated.

        This is a critical function that ensures the state updates
        are properly propagated through the transform chain.
        """
        # Make a deep copy to avoid modifying the original
        tensordict_copy = tensordict.clone()

        # Get the base environment
        base_env = self.env
        while hasattr(base_env, "base_env"):
            base_env = base_env.base_env

        # Step the base environment directly first to see changes
        # Note: We need to map the action through transforms first
        td_for_base = tensordict_copy.clone()

        # Get result from base environment
        result_from_base = base_env._step(td_for_base)

        # Now step through the full transform chain
        result = self.env.step(tensordict_copy)

        # Verify if state changed in the transformed result
        if torch.allclose(result["th"], tensordict_copy["th"]) and torch.allclose(result["thdot"],
                                                                                  tensordict_copy["thdot"]):
            print("WARNING: Transform didn't update state, forcing update from base environment")
            # Force update the critical state variables from the base result
            result["th"] = result_from_base["th"]
            result["thdot"] = result_from_base["thdot"]
            result["phi"] = result_from_base["phi"]
            result["phidot"] = result_from_base["phidot"]

        return result

    def _set_seed(self, seed: Optional[int]):
        """Set random seed for the environment."""
        # Just forward to the wrapped environment
        return self.env.set_seed(seed)

    # Forward any other methods to the wrapped environment
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)


# Alternative solution: Create a direct implementation without transforms
class DirectPendulumEnv(PendulumEnv):
    """
    A direct implementation of the pendulum environment without using transforms.
    This avoids any issues with the transform chain.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Update observation spec to include the concatenated observation
        self.observation_spec = CompositeSpec(
            th=self.observation_spec["th"],
            thdot=self.observation_spec["thdot"],
            phi=self.observation_spec["phi"],
            phidot=self.observation_spec["phidot"],
            sin_th=BoundedTensorSpec(low=-1, high=1, shape=()),
            cos_th=BoundedTensorSpec(low=-1, high=1, shape=()),
            sin_phi=BoundedTensorSpec(low=-1, high=1, shape=()),
            cos_phi=BoundedTensorSpec(low=-1, high=1, shape=()),
            observation=BoundedTensorSpec(low=-1, high=1, shape=(6,)),
            params=self.observation_spec["params"],
            shape=(),
        )

    def _step(self, tensordict):
        # Process the base step
        result = super()._step(tensordict)

        # Add the transformed observations
        self._add_transformed_obs(result)

        return result

    def _reset(self, tensordict=None, **kwargs):
        # Process the base reset
        result = super()._reset(tensordict, **kwargs)

        # Add the transformed observations
        self._add_transformed_obs(result)

        return result

    def _add_transformed_obs(self, tensordict):
        """Helper method to add transformed observations to a tensordict."""
        # Calculate sin and cos transforms
        tensordict["sin_th"] = torch.sin(tensordict["th"])
        tensordict["cos_th"] = torch.cos(tensordict["th"])
        tensordict["sin_phi"] = torch.sin(tensordict["phi"])
        tensordict["cos_phi"] = torch.cos(tensordict["phi"])

        # Create the concatenated observation
        # Unsqueeze for proper dimensions if needed
        sin_th = tensordict["sin_th"].unsqueeze(-1) if tensordict["sin_th"].ndim == tensordict["th"].ndim else \
        tensordict["sin_th"]
        cos_th = tensordict["cos_th"].unsqueeze(-1) if tensordict["cos_th"].ndim == tensordict["th"].ndim else \
        tensordict["cos_th"]
        thdot = tensordict["thdot"].unsqueeze(-1) if tensordict["thdot"].ndim == tensordict["th"].ndim else tensordict[
            "thdot"]
        sin_phi = tensordict["sin_phi"].unsqueeze(-1) if tensordict["sin_phi"].ndim == tensordict["phi"].ndim else \
        tensordict["sin_phi"]
        cos_phi = tensordict["cos_phi"].unsqueeze(-1) if tensordict["cos_phi"].ndim == tensordict["phi"].ndim else \
        tensordict["cos_phi"]
        phidot = tensordict["phidot"].unsqueeze(-1) if tensordict["phidot"].ndim == tensordict["phi"].ndim else \
        tensordict["phidot"]

        # Concatenate observations
        tensordict["observation"] = torch.cat([
            sin_th, cos_th, thdot, sin_phi, cos_phi, phidot
        ], dim=-1)

        return tensordict


# Modified setup function to use the direct environment
def setup_direct_environment():
    """
    Set up the pendulum environment using the direct implementation.

    Returns:
        DirectPendulumEnv: The environment ready for training
    """
    env = DirectPendulumEnv()

    # Verify environment specs
    check_env_specs(env)

    return env


# Use this function to create the fixed environment
def create_fixed_environment():
    """
    Create an environment with our special wrapper to fix state update issues.

    Returns:
        FixedTransformedEnv: A wrapped environment that correctly handles state updates
    """
    # First create the standard environment with transforms
    env = setup_environment()

    # Wrap it with our special wrapper
    fixed_env = FixedTransformedEnv(env)

    return fixed_env


# This is a completely standalone solution that bypasses the transform issues
# with a direct manual rollout.

def manual_debug_policy(policy, env, steps=10):
    """Debug policy behavior by directly accessing the base environment."""
    # Get base environment (the one without transforms)
    base_env = env
    while hasattr(base_env, "base_env"):
        base_env = base_env.base_env

    # Reset environment
    td = base_env.reset()

    # Helper function to safely get item values from tensors
    def safe_item(tensor):
        if tensor.ndim > 0 and tensor.size(0) > 1:
            return tensor[0].item()
        return tensor.item()

    print("Initial state:",
          f"th={safe_item(td['th']):.4f}",
          f"thdot={safe_item(td['thdot']):.4f}",
          f"phi={safe_item(td['phi']):.4f}",
          f"phidot={safe_item(td['phidot']):.4f}")

    print("\nPolicy outputs and state transitions:")
    for i in range(steps):
        # Create an expanded TensorDict with all the transformed observations needed for policy
        full_td = td.clone()

        # Add sin/cos transforms manually
        full_td["sin_th"] = torch.sin(full_td["th"])
        full_td["cos_th"] = torch.cos(full_td["th"])
        full_td["sin_phi"] = torch.sin(full_td["phi"])
        full_td["cos_phi"] = torch.cos(full_td["phi"])

        # Handle dimensionality for concatenation
        sin_th = full_td["sin_th"].unsqueeze(-1) if full_td["sin_th"].ndim == full_td["th"].ndim else full_td["sin_th"]
        cos_th = full_td["cos_th"].unsqueeze(-1) if full_td["cos_th"].ndim == full_td["th"].ndim else full_td["cos_th"]
        thdot = full_td["thdot"].unsqueeze(-1) if full_td["thdot"].ndim == full_td["th"].ndim else full_td["thdot"]
        sin_phi = full_td["sin_phi"].unsqueeze(-1) if full_td["sin_phi"].ndim == full_td["phi"].ndim else full_td[
            "sin_phi"]
        cos_phi = full_td["cos_phi"].unsqueeze(-1) if full_td["cos_phi"].ndim == full_td["phi"].ndim else full_td[
            "cos_phi"]
        phidot = full_td["phidot"].unsqueeze(-1) if full_td["phidot"].ndim == full_td["phi"].ndim else full_td["phidot"]

        # Concatenate observations
        full_td["observation"] = torch.cat([
            sin_th, cos_th, thdot, sin_phi, cos_phi, phidot
        ], dim=-1)

        # Store original state for display
        orig_th = safe_item(td['th'])
        orig_thdot = safe_item(td['thdot'])
        orig_phi = safe_item(td['phi'])
        orig_phidot = safe_item(td['phidot'])

        # Get action from policy using the fully populated tensordict
        with torch.no_grad():
            action_td = policy(full_td.clone())

        # Get scalar action value for display
        action = safe_item(action_td["action"])

        print(f"Step {i}: State=[{orig_th:.4f}, {orig_thdot:.4f}, "
              f"{orig_phi:.4f}, {orig_phidot:.4f}], Action={action:.4f}")

        # Apply action directly to base environment
        td_with_action = td.clone()
        td_with_action["action"] = action_td["action"]

        # Step base environment directly
        td_next = base_env._step(td_with_action)

        # CRITICAL: Replace td with the new state
        td = td_next

        # Print resulting state
        print(f"  → Next state: [{safe_item(td['th']):.4f}, {safe_item(td['thdot']):.4f}, "
              f"{safe_item(td['phi']):.4f}, {safe_item(td['phidot']):.4f}]")

    return td


def manual_train_policy(env, policy, iterations=500, batch_size=32):
    """
    Train a policy on the pendulum environment using manual rollouts.

    This completely bypasses the environment's step and rollout methods,
    working directly with the base environment.

    Args:
        env: The environment to train on
        policy: The policy to train
        iterations: Number of training iterations
        batch_size: Batch size for training

    Returns:
        policy: The trained policy
        logs: Training logs
    """
    # Get the base environment (without transforms)
    base_env = env
    while hasattr(base_env, "base_env"):
        base_env = base_env.base_env

    # Create exploration policy
    exploration_policy = ExplorationPolicy(policy, noise_scale=1.0)

    # Create optimizer and scheduler
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, batch_size * iterations)

    # Set up training loop
    pbar = tqdm.tqdm(range(iterations))
    logs = defaultdict(list)
    best_return = float('-inf')
    patience = 0
    max_patience = 20
    noise_scale = 1.0

    for i in pbar:
        # Reduce exploration noise over time
        if i % 100 == 0 and noise_scale > 0.1:
            noise_scale *= 0.9
            exploration_policy.noise_scale = noise_scale

        # Process each batch element
        all_returns = []
        all_final_rewards = []

        # Generate parameters
        param_td = base_env.gen_params(batch_size=[batch_size])

        for b in range(batch_size):
            # Extract single element parameters
            batch_params = TensorDict({
                "params": TensorDict({
                    k: v[b:b + 1] if v.ndim > 0 else v
                    for k, v in param_td["params"].items()
                }, [])
            }, [])

            # Reset base environment
            td = base_env._reset(batch_params)

            # Initialize accumulators
            rewards = []

            # Perform manual rollout for 100 steps
            for step in range(100):
                # Create the observation needed by the policy
                full_td = td.clone()

                # Add sin/cos transforms manually
                full_td["sin_th"] = torch.sin(full_td["th"])
                full_td["cos_th"] = torch.cos(full_td["th"])
                full_td["sin_phi"] = torch.sin(full_td["phi"])
                full_td["cos_phi"] = torch.cos(full_td["phi"])

                # Handle dimensionality for concatenation
                sin_th = full_td["sin_th"].unsqueeze(-1) if full_td["sin_th"].ndim == full_td["th"].ndim else full_td[
                    "sin_th"]
                cos_th = full_td["cos_th"].unsqueeze(-1) if full_td["cos_th"].ndim == full_td["th"].ndim else full_td[
                    "cos_th"]
                thdot = full_td["thdot"].unsqueeze(-1) if full_td["thdot"].ndim == full_td["th"].ndim else full_td[
                    "thdot"]
                sin_phi = full_td["sin_phi"].unsqueeze(-1) if full_td["sin_phi"].ndim == full_td["phi"].ndim else \
                full_td["sin_phi"]
                cos_phi = full_td["cos_phi"].unsqueeze(-1) if full_td["cos_phi"].ndim == full_td["phi"].ndim else \
                full_td["cos_phi"]
                phidot = full_td["phidot"].unsqueeze(-1) if full_td["phidot"].ndim == full_td["phi"].ndim else full_td[
                    "phidot"]

                # Concatenate observations
                full_td["observation"] = torch.cat([
                    sin_th, cos_th, thdot, sin_phi, cos_phi, phidot
                ], dim=-1)

                # Get action from exploration policy
                with torch.no_grad():
                    action_td = exploration_policy(full_td.clone())

                # Add action to state
                td_with_action = td.clone()
                td_with_action["action"] = action_td["action"]

                # Step environment directly
                next_td = base_env._step(td_with_action)

                # Store reward
                rewards.append(next_td["reward"])

                # Update state
                td = next_td

            # Calculate return for this batch element
            batch_return = torch.stack(rewards).mean()
            all_returns.append(batch_return)
            all_final_rewards.append(rewards[-1])

        # Compute average return across batch
        traj_return = torch.mean(torch.stack(all_returns))
        last_reward = torch.mean(torch.stack(all_final_rewards))

        # Compute loss and update policy
        (-traj_return).backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

        optim.step()
        optim.zero_grad()

        # Track best model
        if traj_return > best_return:
            best_return = traj_return
            # Save best model
            torch.save(policy.state_dict(), "best_pendulum_policy.pt")
            patience = 0
        else:
            patience += 1

        # Early stopping
        if patience > max_patience:
            print(f"Early stopping at iteration {i}")
            # Load best model
            policy.load_state_dict(torch.load("best_pendulum_policy.pt"))
            break

        # Update progress bar
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {last_reward: 4.4f}, "
            f"noise: {noise_scale:.3f}, grad norm: {grad_norm: 4.4}"
        )

        # Log metrics
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(last_reward.item())

        # Step scheduler
        scheduler.step()

        # Occasionally debug policy
        if i % 50 == 0:
            manual_debug_policy(policy, env)

    return policy, logs


def manual_test_environment_step(env):
    """Test if the base environment step function is working correctly."""
    # Get base environment
    base_env = env
    while hasattr(base_env, "base_env"):
        base_env = base_env.base_env

    # Reset environment
    td = base_env.reset()

    print(f"Initial state: th={td['th'].item():.4f}, thdot={td['thdot'].item():.4f}")

    # Apply a constant action
    action_value = 1.0
    td["action"] = torch.tensor([action_value], device=td.device)

    # Step directly with the base environment's _step method
    new_td = base_env._step(td)

    print(f"After base _step: th={new_td['th'].item():.4f}, thdot={new_td['thdot'].item():.4f}")

    # Check if the state is being updated
    if td['th'].item() != new_td['th'].item():
        print("Base _step is correctly updating the state ✓")
    else:
        print("Base _step is NOT updating the state ✗")


def main():
    """
    Main function to run the pendulum simulation.
    """
    # Set up standard environment - we'll use the base env directly
    env = setup_environment()

    # Create policy
    policy = create_policy_network(env)

    # Test base environment step function
    manual_test_environment_step(env)

    # Train policy with manual rollouts
    trained_policy, logs = manual_train_policy(env, policy)

    # Plot training results
    plot_training_results(logs)

    # Debug the trained policy
    print("\nDebug of trained policy:")
    manual_debug_policy(trained_policy, env, steps=20)

    # Evaluate and visualize policy - implement a manual version if needed
    # animation, history = evaluate_and_visualize(trained_policy, env, save_path="pendulum_animation.gif")

    print("Training complete!")


if __name__ == "__main__":
    main()