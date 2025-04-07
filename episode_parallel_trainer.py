import multiprocessing as mp
import numpy as np
import torch
import os
import json
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt


from SimRL import PendulumEnv, SACAgent, ReplayBuffer, plot_training_episode, normalize_angle


class EpisodeParallelTrainer:
    """
    Manages parallel training at the episode level with evolutionary hyperparameter selection.

    In this implementation, we maintain a single network architecture for all workers,
    but train them with different hyperparameters in parallel for each episode.
    After each episode, we select the best-performing hyperparameters to carry forward.
    """

    def __init__(
            self,
            num_workers=3,
            max_episodes=500,
            hidden_dim=256,
            state_dim=6,
            action_dim=1,
            output_dir="episode_parallel_results"
    ):
        """
        Initialize the episode-level parallel trainer.

        Args:
            num_workers: Number of parallel processes for each episode
            max_episodes: Total number of episodes to train
            hidden_dim: Fixed hidden dimension size for all networks
            state_dim: State dimension
            action_dim: Action dimension
            output_dir: Directory to save results
        """
        self.num_workers = num_workers
        self.max_episodes = max_episodes
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Tracking the best hyperparameters
        self.best_hyperparams = None
        self.best_reward = -float('inf')

        # Base agent with consistent architecture - we'll clone this for each worker
        # but with different hyperparameters
        self.base_agent = None

        # Results tracking
        self.episode_results = []

        # Replay buffer - shared across episodes for continuity
        self.replay_buffer = None

        # Environment parameters - will change each episode
        self.current_env_params = None

        # Ensure reproducibility
        self.base_seed = 42

    def initialize_base_agent(self):
        """Initialize the base agent with consistent architecture."""
        # Create agent with default hyperparameters - we'll modify these for workers
        self.base_agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            max_episodes=self.max_episodes
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(100000)

        return self.base_agent

    def generate_hyperparameters(self, include_best=False):
        """
        Generate a list of hyperparameter sets for the episode.
        If include_best is True and we have best_hyperparams, include it as the first set.
        """
        hyperparams_list = []

        # Include the best hyperparameters from previous episode if available
        if include_best and self.best_hyperparams is not None:
            hyperparams_list.append(deepcopy(self.best_hyperparams))

        # Generate random hyperparameters for the remaining workers
        remaining = self.num_workers - len(hyperparams_list)
        for i in range(remaining):
            # Generate random hyperparameters with reasonable ranges
            # Note: hidden_dim is NOT included since all workers use the same architecture
            params = {
                # Learning rates (log scale between 1e-4 and 5e-3)
                'lr': 10 ** np.random.uniform(-4, -2.3),

                # RL algorithm parameters
                'gamma': np.random.uniform(0.97, 0.995),  # Discount factor
                'tau': np.random.uniform(0.001, 0.01),  # Soft update rate

                # Entropy settings
                'automatic_entropy_tuning': bool(np.random.choice([True, False], p=[0.8, 0.2])),
                'alpha': np.random.uniform(0.05, 0.3),

                # Updates per step
                'updates_per_step': np.random.choice([1, 2, 3]),
            }

            # Add a unique seed for reproducibility
            params['seed'] = self.base_seed + i + 1

            hyperparams_list.append(params)

        return hyperparams_list

    def generate_environment_params(self):
        """Generate random environment parameters for this episode."""
        params = {
            'variable_dt': True,  # Always use variable time steps
            'param_variation': 0.2,  # Random parameter variation
            'fixed_params': False,  # Always vary parameters between episodes
            # Random voltage range (higher upper bound = harder problem)
            'voltage_range': (4.0, 18.0)
        }

        self.current_env_params = params
        return params

    def worker_process(self, worker_id, base_state_dict, hyperparam_queue, result_queue, episode_num):
        """
        Worker process function that trains an agent for one episode.

        Args:
            worker_id: ID of this worker
            base_state_dict: State dict of the base agent to clone (with detached tensors)
            hyperparam_queue: Queue to receive hyperparameters
            result_queue: Queue to send results back
            episode_num: Current episode number
        """
        try:
            # Get hyperparameters and environment params from queue
            hyperparams, env_params, replay_buffer_data = hyperparam_queue.get()

            print(f"Episode {episode_num}, Worker {worker_id} starting with hyperparams: {hyperparams}")

            # Set the seed for reproducibility
            torch.manual_seed(hyperparams.get('seed', 42))
            np.random.seed(hyperparams.get('seed', 42))

            # Create environment with the given parameters
            env = PendulumEnv(**env_params)

            # Create a new agent with the specified hyperparameters but same architecture
            agent = SACAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,  # Same hidden dim for all workers
                lr=hyperparams.get('lr', 3e-4),
                gamma=hyperparams.get('gamma', 0.99),
                tau=hyperparams.get('tau', 0.005),
                alpha=hyperparams.get('alpha', 0.2),
                automatic_entropy_tuning=hyperparams.get('automatic_entropy_tuning', True),
                max_episodes=self.max_episodes
            )

            # Load the base agent state dict to ensure architectural consistency
            agent.actor.load_state_dict(base_state_dict)

            # Reconstruct replay buffer from the data
            replay_buffer = ReplayBuffer(100000)
            if replay_buffer_data:
                for transition in replay_buffer_data:
                    replay_buffer.push(*transition)

            # Train for one episode
            episode_result = self._train_single_episode(
                env=env,
                agent=agent,
                replay_buffer=replay_buffer,
                hyperparams=hyperparams,
                episode_num=episode_num,
                worker_id=worker_id
            )

            # Get the agent's state dict and ensure all tensors are detached
            agent_state_dict = {}
            for key, tensor in agent.actor.state_dict().items():
                agent_state_dict[key] = tensor.detach().cpu().clone()

            # Return the results along with the agent state dict
            result = {
                'worker_id': worker_id,
                'episode_num': episode_num,
                'hyperparams': hyperparams,
                'env_params': env_params,
                'reward': episode_result['episode_reward'],
                'agent_state_dict': agent_state_dict,
                'replay_buffer_data': self._sample_replay_buffer(replay_buffer, max_samples=5000),
                'metrics': episode_result
            }

            result_queue.put(result)
            print(
                f"Episode {episode_num}, Worker {worker_id} completed. Reward: {episode_result['episode_reward']:.2f}")

        except Exception as e:
            import traceback
            print(f"Episode {episode_num}, Worker {worker_id} encountered an error: {str(e)}")
            print(traceback.format_exc())

            # Send error information to the result queue
            result_queue.put({
                'worker_id': worker_id,
                'episode_num': episode_num,
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def _sample_replay_buffer(self, replay_buffer, max_samples=5000):
        """Sample experiences from replay buffer to share between workers."""
        # Return a subset of the replay buffer to avoid excessive serialization
        if len(replay_buffer) == 0:
            return []

        indices = np.random.choice(
            len(replay_buffer),
            min(max_samples, len(replay_buffer)),
            replace=False
        )

        # Create a list of sampled transitions
        transitions = []
        for i in indices:
            transitions.append(replay_buffer.buffer[i])

        return transitions

    def _train_single_episode(self, env, agent, replay_buffer, hyperparams, episode_num, worker_id):
        """
        Train an agent for a single episode.

        Args:
            env: The environment to train in
            agent: The agent to train
            replay_buffer: Replay buffer to use
            hyperparams: Hyperparameters for training
            episode_num: Current episode number
            worker_id: Worker ID for logging
        """
        # Extract hyperparameters
        batch_size = 256  # Fixed batch size
        updates_per_step = hyperparams.get('updates_per_step', 1)

        # Prepare for episode
        state = env.reset()
        episode_reward = 0

        # Track losses for reporting
        critic_losses = []
        actor_losses = []
        alpha_values = []

        # Prepare for episode visualization
        episode_states = []
        episode_actions = []
        step_rewards = []

        # Episode loop - run for the full number of steps
        for step in range(env.max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Store state, action, and reward for visualization
            episode_states.append(env.state.copy())
            episode_actions.append(action)
            step_rewards.append(reward)

            # Move to next state
            state = next_state
            episode_reward += reward

            # Update networks if enough samples
            if len(replay_buffer) > batch_size:
                for _ in range(updates_per_step):
                    update_info = agent.update_parameters(replay_buffer, batch_size)
                    critic_losses.append(update_info['critic_loss'])
                    actor_losses.append(update_info['actor_loss'])
                    alpha_values.append(update_info['alpha'])

            if done:
                break

        # Calculate average losses
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_alpha = np.mean(alpha_values) if alpha_values else 0.0

        # Create plots for visualization
        if episode_states and len(episode_states) > 0:
            plot_path = os.path.join(self.output_dir, f"episode_{episode_num}_worker_{worker_id}.png")
            plot_training_episode(
                episode_num,
                np.array(episode_states),
                np.array(episode_actions),
                env.time_history if env.variable_dt else [env.dt] * len(episode_states),
                episode_reward,
                env.params['max_voltage'],
                step_rewards,
                save_path=plot_path
            )

        return {
            'episode_reward': episode_reward,
            'critic_loss': avg_critic_loss,
            'actor_loss': avg_actor_loss,
            'alpha': avg_alpha,
            'steps': step + 1,
            'max_voltage': env.params['max_voltage']
        }

    def run(self):
        """Run the episode-level parallel training process."""
        start_time = time()

        # Initialize the base agent with consistent architecture
        base_agent = self.initialize_base_agent()

        # Prepare to track rewards
        all_rewards = []
        best_rewards = []

        for episode in range(self.max_episodes):
            print(f"\n=== Starting Episode {episode + 1}/{self.max_episodes} ===")

            # Generate environment parameters for this episode
            env_params = self.generate_environment_params()
            print(f"Environment params: {env_params}")

            # Generate hyperparameters for each worker (include best from previous episode)
            hyperparams_list = self.generate_hyperparameters(include_best=(episode > 0))

            # Create queues for communication with workers
            hyperparam_queue = mp.Queue()
            result_queue = mp.Queue()

            # Sample the replay buffer
            replay_buffer_sample = self._sample_replay_buffer(self.replay_buffer) if self.replay_buffer else []

            # Put hyperparameters, environment params, and sampled replay buffer in queue for each worker
            for hyperparams in hyperparams_list:
                hyperparam_queue.put((hyperparams, env_params, replay_buffer_sample))

            # Get the base agent's state dict and DETACH all tensors
            base_state_dict = {}
            for key, tensor in base_agent.actor.state_dict().items():
                base_state_dict[key] = tensor.detach().cpu().clone()

            # Start worker processes
            processes = []
            for worker_id in range(self.num_workers):
                p = mp.Process(
                    target=self.worker_process,
                    args=(worker_id, base_state_dict, hyperparam_queue, result_queue, episode)
                )
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Collect results
            episode_results = []
            for _ in range(self.num_workers):
                if not result_queue.empty():
                    result = result_queue.get()
                    if 'error' not in result:
                        episode_results.append(result)
                    else:
                        print(f"Worker {result['worker_id']} failed with error: {result['error']}")

            # Find the best performer in this episode
            if episode_results:
                # Sort by episode reward (higher is better)
                episode_results.sort(key=lambda x: x['reward'], reverse=True)
                best_result = episode_results[0]

                print(f"\nBest performer: Worker {best_result['worker_id']}")
                print(f"Episode reward: {best_result['reward']:.2f}")
                print(f"Hyperparameters: {best_result['hyperparams']}")

                # Track all rewards and best reward
                episode_rewards = [r['reward'] for r in episode_results]
                all_rewards.append(episode_rewards)
                best_rewards.append(best_result['reward'])

                # Update overall best if this is better
                if best_result['reward'] > self.best_reward:
                    self.best_reward = best_result['reward']
                    self.best_hyperparams = deepcopy(best_result['hyperparams'])

                    # Update the base agent to use the best agent's state dict
                    base_agent.actor.load_state_dict(best_result['agent_state_dict'])

                    print(f"New best performance found! Reward: {self.best_reward:.2f}")
                else:
                    print(f"No improvement over previous best: {self.best_reward:.2f}")

                # Initialize replay buffer if needed
                if self.replay_buffer is None:
                    self.replay_buffer = ReplayBuffer(100000)

                # Get replay buffer data from best performer and add to our buffer
                if 'replay_buffer_data' in best_result and best_result['replay_buffer_data']:
                    for transition in best_result['replay_buffer_data']:
                        self.replay_buffer.push(*transition)

            # Store results for this episode
            self.episode_results.append(episode_results)

            # Plot progress so far
            self._plot_training_progress(best_rewards, all_rewards, episode)

            # Save intermediate state
            self.save_results(os.path.join(self.output_dir, f"results_ep{episode + 1}.json"))

        # Training complete
        total_time = time() - start_time
        print(f"\n=== Training Complete ===")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best reward: {self.best_reward:.2f}")
        print(f"Best hyperparameters: {self.best_hyperparams}")

        # Save final results
        self.save_results(os.path.join(self.output_dir, "results_final.json"))

        # Final evaluation of the best agent
        self._final_evaluation(base_agent)

        return base_agent, self.best_hyperparams, self.best_reward

    def _plot_training_progress(self, best_rewards, all_rewards, episode):
        """Plot training progress across episodes."""
        # Plot best rewards over episodes
        plt.figure(figsize=(12, 8))

        # Plot 1: Best reward for each episode
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(best_rewards) + 1), best_rewards, 'b-', marker='o')
        plt.ylabel('Best Reward per Episode')
        plt.title('Training Progress')
        plt.grid(True)

        # Plot 2: Box plot of rewards for all workers
        plt.subplot(2, 1, 2)
        plt.boxplot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Worker Rewards')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"training_progress_ep{episode + 1}.png"))
        plt.close()

    def _final_evaluation(self, agent, num_episodes=5):
        """Perform final evaluation of the best agent."""
        print("\n=== Final Evaluation of Best Agent ===")

        eval_dir = os.path.join(self.output_dir, "final_evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        # Different evaluation scenarios
        eval_scenarios = [
            {"name": "standard", "variable_dt": True, "param_variation": 0.0,
             "voltage_range": (4.0, 4.0), "num_episodes": num_episodes},
            {"name": "low_variation", "variable_dt": True, "param_variation": 0.1,
             "voltage_range": (4.0, 8.0), "num_episodes": num_episodes},
            {"name": "med_variation", "variable_dt": True, "param_variation": 0.2,
             "voltage_range": (4.0, 12.0), "num_episodes": num_episodes},
            {"name": "high_variation", "variable_dt": True, "param_variation": 0.3,
             "voltage_range": (4.0, 16.0), "num_episodes": num_episodes}
        ]

        for scenario in eval_scenarios:
            scenario_name = scenario.pop("name")
            scenario_dir = os.path.join(eval_dir, scenario_name)
            os.makedirs(scenario_dir, exist_ok=True)

            print(f"\n--- {scenario_name.replace('_', ' ').title()} Evaluation ---")

            self._evaluate_scenario(agent, scenario, scenario_dir)

    def _evaluate_scenario(self, agent, scenario_params, output_dir):
        """Evaluate the agent on a specific scenario."""
        env = PendulumEnv(**scenario_params)

        # Results tracking
        rewards = []
        balance_times = []

        for episode in range(scenario_params["num_episodes"]):
            state = env.reset(random_init=False)  # Start from standard position
            total_reward = 0

            # Data collection for visualization
            states_history = []
            actions_history = []
            step_rewards = []

            for step in range(env.max_steps):
                # Select action without exploration
                action = agent.select_action(state, evaluate=True)

                # Perform action
                next_state, reward, done, _ = env.step(action)

                # Record data
                states_history.append(env.state.copy())
                actions_history.append(action)
                step_rewards.append(reward)

                total_reward += reward
                state = next_state

                if done:
                    break

            rewards.append(total_reward)
            print(f"Evaluation episode {episode + 1}: Reward = {total_reward:.2f}")

            # Calculate balance metrics
            if len(states_history) > 0:
                states_history = np.array(states_history)
                alphas = states_history[:, 1]  # theta_1 (pendulum angle)

                # Normalize angles
                alpha_normalized = np.array([normalize_angle(a + np.pi) for a in alphas])

                # Calculate balance time
                balanced_time = 0.0
                upright_threshold = 0.17  # about 10 degrees

                for i in range(len(alpha_normalized)):
                    if abs(alpha_normalized[i]) < upright_threshold:
                        dt_value = env.time_history[i] if env.variable_dt and i < len(env.time_history) else env.dt
                        balanced_time += dt_value

                balance_times.append(balanced_time)
                print(f"  Time balanced: {balanced_time:.2f} seconds")

            # Create evaluation plot
            plot_path = os.path.join(output_dir, f"eval_ep{episode + 1}.png")
            if len(states_history) > 0:
                plot_training_episode(
                    episode,
                    np.array(states_history),
                    np.array(actions_history),
                    env.time_history if env.variable_dt else [env.dt] * len(states_history),
                    total_reward,
                    env.params['max_voltage'],
                    step_rewards,
                    is_eval=True,
                    save_path=plot_path
                )

        # Print summary
        print("\n--- Scenario Evaluation Summary ---")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        if balance_times:
            print(f"Average balanced time: {np.mean(balance_times):.2f} ± {np.std(balance_times):.2f} seconds")

    def save_results(self, filename):
        """Save training results to a JSON file."""

        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj

        # Extract relevant results, excluding the actual replay buffer data and state dicts
        simplified_episode_results = []
        for episode_data in self.episode_results:
            simplified_data = []
            for worker_data in episode_data:
                worker_data_copy = worker_data.copy()
                # Remove large binary data
                if 'agent_state_dict' in worker_data_copy:
                    worker_data_copy.pop('agent_state_dict')
                if 'replay_buffer_data' in worker_data_copy:
                    worker_data_copy.pop('replay_buffer_data')
                simplified_data.append(worker_data_copy)
            simplified_episode_results.append(simplified_data)

        results = {
            'best_reward': float(self.best_reward),
            'best_hyperparams': convert_to_serializable(self.best_hyperparams),
            'hidden_dim': self.hidden_dim,
            'num_workers': self.num_workers,
            'max_episodes': self.max_episodes,
            'episode_results': convert_to_serializable(simplified_episode_results)
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)