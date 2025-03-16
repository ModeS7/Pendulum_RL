
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


def _step(tensordict):
    th, thdot = tensordict["th"], tensordict["thdot"]
    phi, phidot = tensordict["phi"], tensordict["phidot"]

    g = tensordict["params", "g"] # Gravity constant (m/s²)
    Mp = tensordict["params", "m"]  # Pendulum mass (kg)
    Lp = tensordict["params", "l"]  # Pendulum length from pivot to center of mass (m) (0.085 + 0.129)/2
    dt = tensordict["params", "dt"]

    u = tensordict["action"].squeeze(-1)
    #u = u.clamp(-tensordict["params", "max_voltage"], tensordict["params", "max_voltage"])
    u = u * 5 # Scale the action

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
    Jr = Jm + Jh + Mr * Lr ** 2 / 3  # Assuming arm is like a rod pivoting at one end

    # Define the derivative function.
    # Given state (th, thdot, phi, phidot), returns their time-derivatives.

    # Motor current and torque (u is held constant during the RK4 step)
    im = (u - km * phidot) / Rm
    tau = kt * im

    # Inertia matrix elements
    M11 = Jr + Mp * Lr ** 2
    M12 = Mp * Lr * Lp / 2 * torch.cos(th)
    M21 = M12
    M22 = Jp
    det_M = M11 * M22 - M12 * M21

    # Coriolis and gravitational (plus damping) terms
    C1 = -Mp * Lr * (Lp / 2) * thdot ** 2 * torch.sin(th) - Br * phidot
    C2 = Mp * g * (Lp / 2) * torch.sin(th) - Bp * thdot
    B1 = tau
    B2 = 0

    # Solve for accelerations
    phidotdot = (M22 * (B1 + C1) - M12 * (B2 + C2)) / det_M
    thdotdot  = (M11 * (B2 + C2) - M21 * (B1 + C1)) / det_M

    # Update the state with RK4 weighted average
    new_thdot = thdot + thdotdot * dt
    new_th    = th    + new_thdot * dt
    new_phidot= phidot+ phidotdot * dt
    new_phi   = phi   + new_phidot * dt

    #new_thdotdot = new_thdotdot.clamp(-torch.pi, torch.pi)
    #new_thdot = (thdot + new_thdotdot * dt).clamp(-10.0, 10.0)

    #new_phi = new_phi.clamp(-tensordict["params", "max_phi"], tensordict["params", "max_phi"])

    costs = (angle_normalize(th) ** 2.0 + 0.1 * thdot ** 2.0 + 0.001 * (u ** 2.0)) * 0.1

    # Phi angle penalty to reward
    #costs += 0.1 * angle_normalize(phi) ** 2

    # Add action variance penalty to reward
    #costs -= 0.1 * ((u - u.mean()) ** 2.0).mean()

    costs -= 0.05 * torch.abs(u)

    costs = torch.clamp(costs, 0.0, 10.0)

    """angle_cost = angle_normalize(th) ** 2.0  # Penalize angle deviation from upright
    velocity_cost = 0.1 * thdot ** 2.0  # Penalize angular velocity
    action_cost = 0.001 * (u ** 2.0)  # Small penalty for action magnitude
    motor_angle_cost = 0.1 * angle_normalize(phi) ** 2.0  # Penalize motor angle deviation

    # Combine costs and scale
    costs = (angle_cost + velocity_cost + action_cost + motor_angle_cost) * 0.1
    costs = torch.clamp(costs, 0.0, 10.0)"""

    reward = -costs.view(*tensordict.shape, 1)
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

def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

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
    )
    thdot = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_thdot - low_thdot)
        + low_thdot
    )
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
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        th=UnboundedContinuousTensorSpec(
            shape=(),
            dtype=torch.float32,
        ),
        thdot=UnboundedContinuousTensorSpec(
            shape=(),
            dtype=torch.float32,
        ),
        phi=UnboundedContinuousTensorSpec(
            shape=(),
            dtype=torch.float32,
        ),
        phidot=UnboundedContinuousTensorSpec(
            shape=(),
            dtype=torch.float32,
        ),

        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )

    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    """self.action_spec = BoundedTensorSpec(
        low=-td_params["params", "max_voltage"],
        high=td_params["params", "max_voltage"],
        shape=(1,),
        dtype=torch.float32,
    )"""
    self.action_spec = UnboundedContinuousTensorSpec(
        shape=(),
        dtype=torch.float32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
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
    # Set the global seed
    torch.manual_seed(seed)
    # Create a device-specific generator
    if self.device.type == 'cuda':
        # For CUDA device
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
    else:
        # For CPU device
        self.rng = torch.manual_seed(seed)


def gen_params(g=9.81, batch_size=None, device=None) -> TensorDictBase:
    if batch_size is None:
        batch_size = []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    td = TensorDict(
        {
            "params": TensorDict(
                {
                    #"max_voltage": torch.tensor(1.0, device=device),
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
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

class PendulumEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
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
#print("observation_spec:", env.observation_spec)
#print("state_spec:", env.state_spec)
#print("reward_spec:", env.reward_spec)
td = env.reset()
#print("reset tensordict", td)
td = env.rand_step(td)
#print("random step tensordict", td)
env = TransformedEnv(
    env,
    # ``Unsqueeze`` the observations that we will concatenate
    UnsqueezeTransform(
        dim=-1,
        in_keys=["th", "thdot", "phi", "phidot"],
        in_keys_inv=["th", "thdot", "phi", "phidot"],
    ),
)

class SinTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.sin()

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
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

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
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
    # preallocate:
    data = TensorDict({}, [steps])
    # reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data


#print("data from rollout:", simple_rollout(100))

batch_size = 10  # number of environments to be executed in batch
td = env.reset(env.gen_params(batch_size=[batch_size]))
#print("reset (batch size of 10)", td)
td = env.rand_step(td)
#print("rand step (batch size of 10)", td)

rollout = env.rollout(
    6,
    auto_reset=False,  # we're executing the reset out of the ``rollout`` call
    tensordict=env.reset(env.gen_params(batch_size=[batch_size])),
)
#print("rollout of len 3 (batch size of 10):", rollout)

torch.manual_seed(0)
env.set_seed(0)


# Get the device from the environment
device = env.device

# Define the network
net = nn.Sequential(
    nn.Linear(6, 256),  # Wider first layer
    nn.ReLU(),
    nn.Linear(256, 256),  # Wider second layer
    nn.ReLU(),
    nn.Linear(256, 128),  # Additional layer for more expressivity
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Tanh(),  # Tanh activation for bounded output
).to(device)

# Initialize weights properly
for m in net.modules():
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2.5))
        nn.init.uniform_(m.bias, -0.01, 0.01)


class ExplorationPolicy(TensorDictModule):
    def __init__(self, policy, noise_scale=0.5):
        # Create a dummy module that will be overridden by our custom forward
        dummy_module = nn.Identity()
        super().__init__(
            module=dummy_module,  # This was the missing argument
            in_keys=policy.in_keys,
            out_keys=policy.out_keys
        )
        self.policy = policy
        self.noise_scale = noise_scale

    def forward(self, tensordict):
        # Get the base action from the policy
        action_td = self.policy(tensordict)

        # Add exploration noise
        noise = torch.randn_like(action_td["action"]) * self.noise_scale
        action_td["action"] = action_td["action"] + noise

        return action_td

policy = TensorDictModule(
    net,
    in_keys=["observation"],
    out_keys=["action"],
)
def debug_policy(policy, env, steps=10):
    # Reset environment
    td = env.reset()

    print("Initial state:",
          f"th={td['th'].item():.4f}",
          f"thdot={td['thdot'].item():.4f}",
          f"phi={td['phi'].item():.4f}",
          f"phidot={td['phidot'].item():.4f}")

    print("\nPolicy outputs and state transitions:")
    for i in range(steps):
        # Get action from policy
        with torch.no_grad():
            action_td = policy(td.clone())
            action = action_td["action"].item()

        print(f"Step {i}: State=[{td['th'].item():.4f}, {td['thdot'].item():.4f}, "
              f"{td['phi'].item():.4f}, {td['phidot'].item():.4f}], Action={action:.4f}")

        # Method 1: Direct step function (this should work)
        # Apply policy action
        td_copy = td.clone()
        td_copy["action"] = action_td["action"]

        # Call _step directly instead of env.step
        new_td = _step(td_copy)

        # Update the original tensordict with new values
        for key in ['th', 'thdot', 'phi', 'phidot', 'reward', 'done']:
            if key in new_td:
                td[key] = new_td[key]

        # Print resulting state
        print(f"  → Next state: [{td['th'].item():.4f}, {td['thdot'].item():.4f}, "
              f"{td['phi'].item():.4f}, {td['phidot'].item():.4f}]")


class NormalizeObservation(Transform):
    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=["observation"])

    def _apply_transform(self, obs):
        # Normalize observation to be in a reasonable range
        # This helps the neural network learn more effectively
        return torch.clamp(obs / 3.0, -1.0, 1.0)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


# Add transform to environment
env.append_transform(NormalizeObservation())

# Wrap policy with exploration
exploration_policy = ExplorationPolicy(policy, noise_scale=1.0)

#optim = torch.optim.Adam(policy.parameters(), lr=2e-4)
#optim = torch.optim.RMSprop(policy.parameters(), lr=1e-4)
optim = Lion(policy.parameters(), lr=2e-3, weight_decay=1e-4)
batch_size = 8192*2
iterations = 2000
# Modify your training loop for better stability
pbar = tqdm.tqdm(range(iterations))  # Smaller number of iterations for testing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, batch_size * iterations)
logs = defaultdict(list)
best_return = float('-inf')
patience = 0
max_patience = 50
noise_scale = 1.0
exploration_policy.noise_scale = noise_scale
max_patience = 50

for i in pbar:
    # Reduce exploration noise over time
    if i % 200 == 0 and noise_scale > 0.2:
        noise_scale *= 0.9
        exploration_policy.noise_scale = noise_scale

    # Every 500 iterations, temporarily increase noise to escape local optima
    if i % 500 == 0 and i > 0:
        temp_noise = noise_scale * 2.0
        exploration_policy.noise_scale = temp_noise
        print(f"Temporarily increasing noise to {temp_noise} to escape local minimum")

    # Reset noise after 10 iterations of increased exploration
    if i % 500 == 10 and i > 0:
        exploration_policy.noise_scale = noise_scale
        print(f"Resetting noise to {noise_scale}")

    init_td = env.reset(env.gen_params(batch_size=[batch_size]))

    # Use exploration during training
    rollout = env.rollout(100, exploration_policy, tensordict=init_td, auto_reset=False)

    traj_return = rollout["next", "reward"].mean()
    (-traj_return).backward()

    # Gradient clipping
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

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

    pbar.set_description(
        f"reward: {traj_return: 4.4f}, "
        f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, "
        f"noise: {noise_scale:.3f}, grad norm: {gn: 4.4}"
    )

    logs["return"].append(traj_return.item())
    logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
    #scheduler.step()

    if i % 50 == 0:  # Every 50 iterations
        # Sample a few episodes to track
        debug_size = min(3, batch_size)
        debug_td = rollout[:debug_size, :5]  # First 3 batches, first 5 timesteps

        print(f"\n--- Iteration {i} State Transitions ---")
        for b in range(debug_size):
            print(f"Batch {b}:")
            for t in range(4):  # Look at transitions between timesteps
                print(f"  t{t} → t{t + 1}: "
                      f"th: {debug_td[b, t]['th'].item():.4f} → {debug_td[b, t + 1]['th'].item():.4f}, "
                      f"thdot: {debug_td[b, t]['thdot'].item():.4f} → {debug_td[b, t + 1]['thdot'].item():.4f}, "
                      f"phi: {debug_td[b, t]['phi'].item():.4f} → {debug_td[b, t + 1]['phi'].item():.4f}, "
                      f"phidot: {debug_td[b, t]['phidot'].item():.4f} → {debug_td[b, t + 1]['phidot'].item():.4f}, "
                      f"action: {debug_td[b, t]['action'].item():.4f}, "
                      f"reward: {debug_td[b, t]['next', 'reward'].item():.4f}")
        debug_policy(policy, env)
debug_policy(policy, env)

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


def visualize_policy(policy, env, steps=100, save_path=None):
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

        # Update plots - only use as many points as we have for each time step
        theta_line.set_data(range(i + 1), history['theta'][:i + 1])
        phi_line.set_data(range(i + 1), history['phi'][:i + 1])

        # Handle the action plot separately (we have one fewer action than theta/phi)
        if i < len(history['action']):
            action_line.set_data(range(len(history['action'][:i + 1])), history['action'][:i + 1])
        else:
            # For the last frame, use all actions
            action_line.set_data(range(len(history['action'])), history['action'])

        return arm_line, pendulum_line, pendulum_bob, theta_line, phi_line, action_line

    # Create animation - use the length of theta as the number of frames
    ani = FuncAnimation(fig, update, frames=len(history['theta']),
                        interval=100,  # Fixed interval for smoother animation
                        blit=True)

    plt.tight_layout()

    # Save if requested
    if save_path:
        print(f"Saving animation to {save_path}...")
        # Use a writer that doesn't require ffmpeg for GIF creation
        writer = 'imagemagick' if save_path.endswith('.gif') else 'pillow'
        try:
            ani.save(save_path, writer=writer, fps=30)
            print("Animation saved!")
        except ValueError as e:
            print(f"Error saving with {writer}, trying different writer...")
            if writer == 'pillow':
                try:
                    ani.save(save_path, writer='imagemagick', fps=30)
                    print("Animation saved with imagemagick!")
                except Exception as e:
                    print(f"Failed to save animation: {e}")
                    # Save figures instead as a fallback
                    plt.savefig(save_path.replace('.gif', '.png'))
                    print(f"Saved still image instead.")
            else:
                print(f"Failed to save animation: {e}")
                plt.savefig(save_path.replace('.gif', '.png'))
                print(f"Saved still image instead.")

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
