import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from tqdm import tqdm
import random
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from copy import deepcopy


# DDPG Components

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def add(self, tensordict):
        # Store a transition as a TensorDict
        self.buffer.append(tensordict.detach().cpu())

    def sample(self, batch_size):
        # Sample a batch of transitions
        indices = random.sample(range(len(self.buffer)), min(len(self.buffer), batch_size))
        samples = [self.buffer[i] for i in indices]

        # Stack and convert to device
        batch = torch.stack(samples, dim=0).to(self.device)
        return batch

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class DDPG:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action=1.0,
            hidden_dim=256,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            actor_lr=3e-4,
            critic_lr=3e-4,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.device = device

        # Initialize actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Initialize critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Set hyperparameters
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        # For TensorDictModule compatibility
        self.policy_module = TensorDictModule(
            self.actor,
            in_keys=["observation"],
            out_keys=["action"],
        )

    def select_action(self, tensordict):
        state = tensordict["observation"]
        with torch.no_grad():
            action = self.actor(state)
        return action

    def select_action_with_noise(self, tensordict, noise_scale=0.1):
        state = tensordict["observation"]
        with torch.no_grad():
            action = self.actor(state)
            noise = torch.randn_like(action) * noise_scale
            action = (action + noise).clamp(-self.max_action, self.max_action)
        return action

    def train(self, replay_buffer, batch_size=256):
        # Sample a batch from the replay buffer
        batch = replay_buffer.sample(batch_size)

        # Get the necessary components from batch
        state = batch["observation"]
        action = batch["action"]
        reward = batch["next", "reward"]
        next_state = batch["next", "observation"]
        done = batch["next", "done"]

        # Compute target Q
        with torch.no_grad():
            # Select next actions using the target actor
            next_action = self.actor_target(next_state)

            # Add noise to next action for smoothing (target policy smoothing)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done.float()) * self.discount * target_Q

        # Compute current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }

    def _update_target_networks(self):
        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth"))
        self.critic_target = deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth"))
        self.actor_target = deepcopy(self.actor)


# Training function for DDPG
def train_ddpg(env, state_dim, action_dim, max_action=1.0, max_steps=100000, batch_size=256,
               replay_buffer_size=1000000, start_timesteps=10000, eval_freq=5000,
               exploration_noise=0.1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Initialize DDPG agent
    agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size, device=device)

    # Logging variables
    logs = defaultdict(list)
    evaluations = []
    timestep = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0

    # TensorDictModule wrapper for policy
    policy_module = TensorDictModule(
        lambda obs: agent.select_action_with_noise({"observation": obs}, noise_scale=exploration_noise),
        in_keys=["observation"],
        out_keys=["action"]
    )

    # Reset the environment
    tensordict = env.reset()
    done = False

    # Progress bar
    pbar = tqdm(total=max_steps)
    best_eval_reward = float('-inf')

    while timestep < max_steps:
        episode_timesteps += 1

        # Select action with exploration noise
        if timestep < start_timesteps:
            # Random action for initial exploration
            action = env.action_spec.rand()
            tensordict["action"] = action
        else:
            # DDPG action with exploration noise
            action = agent.select_action_with_noise(tensordict, exploration_noise)
            tensordict["action"] = action

        # Perform action
        next_tensordict = env.step(tensordict)

        # Get reward and done signal
        reward = next_tensordict["reward"]
        done = next_tensordict["done"]

        # Store transition in replay buffer
        transition = TensorDict({
            "observation": tensordict["observation"],
            "action": tensordict["action"],
            "next": {
                "observation": next_tensordict["observation"],
                "reward": next_tensordict["reward"],
                "done": next_tensordict["done"]
            }
        }, batch_size=[])

        replay_buffer.add(transition)

        # Update the current state
        tensordict = TensorDict({
            "th": next_tensordict["th"],
            "thdot": next_tensordict["thdot"],
            "phi": next_tensordict["phi"],
            "phidot": next_tensordict["phidot"],
            "observation": next_tensordict["observation"],
            "params": next_tensordict["params"]
        }, batch_size=[])

        episode_reward += reward.item()
        timestep += 1
        pbar.update(1)

        # Train agent after collecting enough samples
        if timestep >= start_timesteps and len(replay_buffer) > batch_size:
            loss_info = agent.train(replay_buffer, batch_size)
            logs["critic_loss"].append(loss_info["critic_loss"])
            logs["actor_loss"].append(loss_info["actor_loss"])

        # Reset environment if done or max episode length reached
        if done.any() or episode_timesteps >= 1000:
            # Record episode statistics
            logs["episode_reward"].append(episode_reward)
            logs["episode_length"].append(episode_timesteps)

            # Reset episode variables
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # Reset environment
            tensordict = env.reset()
            done = False

            pbar.set_description(f"Episode {episode_num} | Avg. Reward: {np.mean(logs['episode_reward'][-100:]):.2f}")

        # Evaluate the agent
        if timestep % eval_freq == 0:
            eval_reward = evaluate_policy(agent, env, eval_episodes=10)
            evaluations.append(eval_reward)

            print(f"Timestep {timestep} | Evaluation reward: {eval_reward:.2f}")

            # Save the model if it's the best so far
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save("best_ddpg_pendulum")

            # Also save periodically
            if timestep % (eval_freq * 5) == 0:
                agent.save(f"ddpg_pendulum_{timestep}")

                # Plot learning curves
                plot_learning_curves(logs)

    pbar.close()
    return agent, logs


def evaluate_policy(agent, env, eval_episodes=10):
    avg_reward = 0.

    for _ in range(eval_episodes):
        tensordict = env.reset()
        done = False
        episode_steps = 0

        while not done.any() and episode_steps < 1000:
            # Select action without exploration noise
            with torch.no_grad():
                action = agent.select_action(tensordict)
                tensordict["action"] = action

            # Perform action
            next_tensordict = env.step(tensordict)

            # Get reward and done signal
            reward = next_tensordict["reward"]
            done = next_tensordict["done"]

            # Update the current state
            tensordict = TensorDict({
                "th": next_tensordict["th"],
                "thdot": next_tensordict["thdot"],
                "phi": next_tensordict["phi"],
                "phidot": next_tensordict["phidot"],
                "observation": next_tensordict["observation"],
                "params": next_tensordict["params"]
            }, batch_size=[])

            avg_reward += reward.item()
            episode_steps += 1

    avg_reward /= eval_episodes
    return avg_reward


def plot_learning_curves(logs):
    plt.figure(figsize=(15, 10))

    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(logs["episode_reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward")

    # Plot episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(logs["episode_length"])
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode Length")

    # Plot critic loss
    plt.subplot(2, 2, 3)
    plt.plot(logs["critic_loss"])
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Critic Loss")

    # Plot actor loss
    plt.subplot(2, 2, 4)
    plt.plot(logs["actor_loss"])
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Actor Loss")

    plt.tight_layout()
    plt.savefig("ddpg_learning_curves.png")
    plt.close()