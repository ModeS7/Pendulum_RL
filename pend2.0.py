from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from lion_pytorch import Lion
import cv2

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

DEFAULT_X = np.pi
DEFAULT_Y = 1.0
# System Parameters (from the QUBE-Servo 2 manual)
# Motor and Pendulum parameters
Rm = 8.4  # Motor resistance (Ohm)
kt = 0.042  # Motor torque constant (N·m/A)
km = 0.042  # Motor back-EMF constant (V·s/rad)
Jm = 4e-6  # Motor moment of inertia (kg·m²)
Jp = 3.3e-5  # Pendulum moment of inertia (kg·m²)
Mp = 0.024  # Pendulum mass (kg)
Lp = 0.129  # Pendulum length from pivot to center of mass (m) (0.085 + 0.129)/2
Br = 0.001  # Rotary arm viscous damping coefficient (N·m·s/rad)
Bp = 0.0005  # Pendulum viscous damping coefficient (N·m·s/rad)
g_default = 9.81  # Gravity constant (m/s²)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def pendulum_dynamics(state, vm, params):
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
    theta, alpha, theta_dot, alpha_dot = torch.split(state, 1, dim=-1)

    Rm = params["Rm"]
    kt = params["kt"]
    km = params["km"]
    Jm = params["Jm"]
    Jp = params["Jp"]
    Mp = params["Mp"]
    Lp = params["Lp"]
    Br = params["Br"]
    Bp = params["Bp"]
    g = params["g"]

    # Motor current
    im = (vm - km * theta_dot) / Rm

    # Motor torque
    tau = kt * im

    # Equations of motion
    # Inertia matrix elements
    M11 = Jm + Jp * torch.sin(alpha) ** 2
    M12 = Jp * Lp * torch.cos(alpha)
    M21 = M12
    M22 = Jp

    # Coriolis and centrifugal terms
    C1 = Jp * torch.sin(2 * alpha) * theta_dot * alpha_dot / 2 + Br * theta_dot
    C2 = -Jp * torch.sin(alpha) * theta_dot ** 2 / 2 + Bp * alpha_dot

    # Gravity terms
    G1 = 0
    G2 = Mp * g * Lp * torch.sin(alpha)

    # Torque input vector
    B1 = tau
    B2 = 0

    # Solve for accelerations
    det = M11 * M22 - M12 * M21
    theta_ddot = (M22 * (B1 - C1 - G1) - M12 * (B2 - C2 - G2)) / det
    alpha_ddot = (-M21 * (B1 - C1 - G1) + M11 * (B2 - C2 - G2)) / det

    return torch.cat([theta_dot, alpha_dot, theta_ddot, alpha_ddot], dim=-1)


def _step(tensordict):
    th = tensordict["th"]
    thdot = tensordict["thdot"]
    phi = tensordict["phi"]
    phidot = tensordict["phidot"]
    u = tensordict["action"].squeeze(-1)
    params = tensordict["params"]
    dt = params["dt"]
    max_voltage = params["max_voltage"]

    u = u.clamp(-max_voltage, max_voltage)

    state = torch.stack([th, phi, thdot, phidot], dim=-1)
    d_state = pendulum_dynamics(state, u.unsqueeze(-1), params)
    thdot_new, phidot_new, thddot_new, phiddot_new = torch.split(d_state, 1, dim=-1)

    new_th = th + thdot_new * dt
    new_thdot = thdot + thddot_new * dt
    new_phi = phi + phidot_new * dt
    new_phidot = phidot + phiddot_new * dt

    costs = angle_normalize(new_th) ** 2.0 + 0.1 * new_thdot ** 2.0 + 0.001 * (u ** 2.0)
    costs += 0.1 * angle_normalize(new_phi) ** 2
    costs += 1 * ((u - u.mean()) ** 2.0).mean()

    reward = -costs.view(*tensordict.shape, 1)
    done = torch.zeros_like(reward, dtype=torch.bool)

    out = TensorDict(
        {
            "th": new_th,
            "thdot": new_thdot,
            "phi": new_phi,
            "phidot": new_phidot,
            "reward": reward,
            "done": done,
            "params": params,
            "action": tensordict["action"] # Keep action for next step if needed
        },
        tensordict.shape,
    )
    return out


def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        tensordict = self.gen_params(batch_size=self.batch_size)

    high_th = torch.tensor(DEFAULT_X, device=self.device)
    high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
    low_th = -high_th
    low_thdot = -high_thdot

    th = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_th - low_th)
        + low_th
    ).unsqueeze(-1)
    thdot = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_thdot - low_thdot)
        + low_thdot
    ).unsqueeze(-1)
    out = TensorDict(
        {
            "th": th,
            "thdot": thdot,
            "phi": torch.zeros_like(th),
            "phidot": torch.zeros_like(th),
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
        device=self.device,  # Ensure the device matches the spec
    )
    return out
def _make_spec(self, td_params):
    self.observation_spec = CompositeSpec(
        th=UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
        ),
        thdot=UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
        ),
        phi=UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
        ),
        phidot=UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
        ),
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )
    self.state_spec = self.observation_spec.clone()
    self.action_spec = BoundedTensorSpec(
        low=-td_params["params", "max_voltage"],
        high=td_params["params", "max_voltage"],
        shape=(1,),
        dtype=torch.float32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
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
    rng = torch.manual_seed(seed)
    self.rng = rng

def gen_params(g=g_default, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters for the QUBE-Servo 2."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_voltage": torch.tensor(5.0),
                    "dt": torch.tensor(0.05),  # Using the simulation dt from the manual
                    "g": torch.tensor(g),
                    "Rm": torch.tensor(Rm),
                    "kt": torch.tensor(kt),
                    "km": torch.tensor(km),
                    "Jm": torch.tensor(Jm),
                    "Jp": torch.tensor(Jp),
                    "Mp": torch.tensor(Mp),
                    "Lp": torch.tensor(Lp),
                    "Br": torch.tensor(Br),
                    "Bp": torch.tensor(Bp),
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

class PendulumEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=td_params.shape)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed

env = PendulumEnv()
check_env_specs(env)
print("observation_spec:", env.observation_spec)
print("state_spec:", env.state_spec)
print("reward_spec:", env.reward_spec)
td = env.reset()
print("reset tensordict", td)
td = env.rand_step(td)
print("random step tensordict", td)
env = TransformedEnv(
    env,
    UnsqueezeTransform(
        dim=-1,
        in_keys=["th", "thdot", "phi", "phidot"],
        in_keys_inv=["th", "thdot", "phi", "phidot"],
    ),
)

class SinTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.sin()

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


class CosTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.cos()

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


t_sin = SinTransform(in_keys=["th"], out_keys=["sin_th"])
t_cos = CosTransform(in_keys=["th"], out_keys=["cos_th"])
p_sin = SinTransform(in_keys=["phi"], out_keys=["sin_phi"])
p_cos = CosTransform(in_keys=["phi"], out_keys=["cos_phi"])
env.append_transform(t_sin)
env.append_transform(t_cos)
env.append_transform(p_sin)
env.append_transform(p_cos)

cat_transform = CatTensors(
    in_keys=["sin_th", "cos_th", "thdot", "sin_phi", "cos_phi", "phidot"],
    dim=-1, out_key="observation", del_keys=False)
env.append_transform(cat_transform)

check_env_specs(env)

def simple_rollout(steps=100):
    data = TensorDict({}, [steps])
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand(_data.shape)
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data


print("data from rollout:", simple_rollout(100))

batch_size = 10
td = env.reset(env.gen_params(batch_size=[batch_size]))
print("reset (batch size of 10)", td)
td = env.rand_step(td)
print("rand step (batch size of 10)", td)

rollout = env.rollout(
    4,
    auto_reset=False,
    tensordict=env.reset(env.gen_params(batch_size=[batch_size])),
)
print("rollout of len 3 (batch size of 10):", rollout)

torch.manual_seed(0)
env.set_seed(0)


net = nn.Sequential(
    nn.Linear(6, 64),  # Correct input size after concatenation
    nn.ReLU(),
    nn.Linear(64, 1),
)
policy = TensorDictModule(
    net,
    in_keys=["observation"],
    out_keys=["action"],
)

optim = torch.optim.Adam(policy.parameters(), lr=2e-3)
batch_size = 8
pbar = tqdm.tqdm(range(100 // batch_size))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1_000)
logs = defaultdict(list)

for _ in pbar:
    init_td = env.reset(env.gen_params(batch_size=[batch_size]))
    rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
    traj_return = rollout["next", "reward"].mean()
    (-traj_return).backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    pbar.set_description(
        f"reward: {traj_return: 4.4f}, "
        f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
    )
    logs["return"].append(traj_return.item())
    logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
    scheduler.step()


def plot():

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(logs["return"])
    plt.title("returns")
    plt.xlabel("iteration")
    plt.subplot(1, 2, 2)
    plt.plot(logs["last_reward"])
    plt.title("last reward")
    plt.xlabel("iteration")
    plt.show()


plot()


def visualize_policy(policy, env, steps=500, save_path=None):
    # Reset environment
    td = env.reset()

    # Lists to store state history
    history = {
        'theta': [],
        'phi': [],
        'action': []
    }

    # Record initial state
    history['theta'].append(td["th"].item())
    history['phi'].append(td["phi"].item())
    history['action'].append(0.0)

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

        # Get updated state using _step directly
        new_td = _step(td)

        # Update our td with new state values
        for key in new_td.keys():
            td[key] = new_td[key]

        # Record state after step
        history['theta'].append(td["th"].item())
        history['phi'].append(td["phi"].item())

        if td["done"].any():
            break


    # Diagnostic - print state variations
    theta_changes = np.diff(history['theta'])
    phi_changes = np.diff(history['phi'])
    action_changes = np.diff(history['action'])

    print(
        f"Theta changes: min={np.min(theta_changes):.6f}, max={np.max(theta_changes):.6f}, avg={np.mean(np.abs(theta_changes)):.6f}")
    print(
        f"Phi changes: min={np.min(phi_changes):.6f}, max={np.max(phi_changes):.6f}, avg={np.mean(np.abs(phi_changes)):.6f}")
    print(
        f"Action changes: min={np.min(action_changes):.6f}, max={np.max(action_changes):.6f}, avg={np.mean(np.abs(action_changes)):.6f}")

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
    ax2.set_ylim(-5*np.pi, 5*np.pi)
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
        action_line.set_data(range(i + 1), history['action'][:i + 1])

        return arm_line, pendulum_line, pendulum_bob, theta_line, phi_line, action_line

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(history['theta']),
                        interval=100,  # Fixed interval for smoother animation
                        blit=True)

    plt.tight_layout()

    # Save if requested
    if save_path:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer='pillow', fps=30)
        print("Animation saved!")

    return ani, history


# Test the visualization
def evaluate_and_visualize(policy, env, save_path=None):
    """Full evaluation and visualization of a trained policy"""
    # Run visualization
    print("Visualizing trained policy...")
    animation, history = visualize_policy(policy, env, steps=200, save_path=save_path)

    # Collect metrics
    print("Evaluating policy performance...")
    with torch.no_grad():
        td = env.reset()
        rollout = env.rollout(100, policy, tensordict=td, auto_reset=False)

    # Calculate metrics
    avg_reward = rollout["next", "reward"].mean().item()
    total_reward = rollout["next", "reward"].sum().item()

    print(f"Average reward: {avg_reward:.6f}")
    print(f"Total reward: {total_reward:.6f}")

    # Additional diagnostics
    return animation, history

# Usage
if __name__ == "__main__":
    # Get the animation from the tuple
    animation_result, history = evaluate_and_visualize(policy, env, save_path="pendulum_animation.gif")

    # Display in notebook/interactive environment
    try:
        from IPython.display import display

        plt.close()  # Close the current figure to avoid duplicates
        display(animation_result.to_jshtml())
    except ImportError:
        plt.show()  # Fallback to regular display