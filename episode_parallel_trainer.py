import multiprocessing as mp
from multiprocessing import Manager
import numpy as np
import torch
import os
import json
import pickle
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import psutil
from datetime import datetime, timedelta

# Import your pendulum code
from SimRL import PendulumEnv, SACAgent, ReplayBuffer, plot_training_episode, normalize_angle


# Modified worker function that returns results directly instead of saving to disk
def worker_with_affinity(worker_id, params, episode_num, worker_cores, result_queue, verbose=False):
    """Worker function wrapper that sets CPU affinity before executing the actual work"""
    try:
        # Import psutil here to ensure it's available in the subprocess
        import psutil

        # Set CPU affinity for this process
        proc = psutil.Process()
        if verbose:
            print(f"Worker {worker_id} setting CPU affinity to cores: {worker_cores}")
        proc.cpu_affinity(worker_cores)

        # Run the modified worker function that returns results directly
        return worker_training_job(worker_id, params, episode_num, result_queue, verbose)
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in worker_with_affinity: {str(e)}")
        print(error_traceback)
        # Put error in queue
        result_queue.put({
            'worker_id': worker_id,
            'episode_num': episode_num,
            'error': str(e),
            'traceback': error_traceback
        })
        return False


def worker_training_job(worker_id, params, episode_num, result_queue, verbose=False):
    try:
        hyperparams = params['hyperparams']
        env_params = params['env_params']
        episode_seed = params.get('episode_seed', 42 + episode_num)
        base_model_path = params['base_model_path']
        initial_replay_buffer = params.get('replay_buffer', None)  # Get the buffer directly
        hidden_dim = params['hidden_dim']
        state_dim = params['state_dim']
        action_dim = params['action_dim']
        output_dir = params['output_dir']

        if verbose:
            print(f"Episode {episode_num}, Worker {worker_id} starting with hyperparams: {hyperparams}")

        # Set the seed for environment parameters
        np.random.seed(episode_seed)

        # Create environment with controlled randomization
        env = PendulumEnv(**env_params)

        # Call reset() first to initialize the parameters
        state = env.reset()

        # Now the parameters are available
        original_params = env.get_current_parameters().copy()

        # Print the actual voltage only in verbose mode
        if verbose:
            max_voltage = original_params['max_voltage']
            print(f"  Using environment with max_voltage={max_voltage:.2f}V")

        # Override the reset method to ensure consistent parameters
        original_reset = env.reset

        def fixed_reset(*args, **kwargs):
            # Call the original reset to get state
            state = original_reset(*args, **kwargs)
            # Force parameters back to our consistent values
            env.params = original_params.copy()
            return state

        # Replace the reset method
        env.reset = fixed_reset

        # Set the seed for training randomization
        seed = hyperparams.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create a new agent
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=hyperparams.get('lr', 3e-4),
            gamma=hyperparams.get('gamma', 0.99),
            tau=hyperparams.get('tau', 0.005),
            alpha=hyperparams.get('alpha', 0.2),
            automatic_entropy_tuning=hyperparams.get('automatic_entropy_tuning', True),
            max_episodes=1000
        )

        # Load base model if available
        if base_model_path and os.path.exists(base_model_path):
            agent.actor.load_state_dict(torch.load(base_model_path, map_location='cpu'))

        # Initialize replay buffer with full capacity
        replay_buffer = ReplayBuffer(100000)  # Increase buffer size to 100000

        # Load from initial replay buffer if provided
        if initial_replay_buffer is not None:
            for transition in initial_replay_buffer:
                replay_buffer.push(*transition)

        # Train for one episode
        batch_size = 512
        updates_per_step = hyperparams.get('updates_per_step', 1)

        # Create a directory for this worker's output
        worker_dir = os.path.join(output_dir, f"episode_{episode_num}", f"worker_{worker_id}")
        os.makedirs(worker_dir, exist_ok=True)

        # Run a single episode
        state = env.reset()
        episode_reward = 0

        # Track losses for reporting
        critic_losses = []
        actor_losses = []
        alpha_values = []

        # Store states and actions for visualization
        episode_states = []
        episode_actions = []
        step_rewards = []

        # Start timing the episode
        episode_start_time = time()

        # Episode loop
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

        # Calculate episode duration
        episode_duration = time() - episode_start_time

        # Save the model
        model_path = os.path.join(worker_dir, "actor_model.pth")
        torch.save(agent.actor.state_dict(), model_path)

        # Generate plot for this episode
        plot_path = os.path.join(worker_dir, "episode_plot.png")
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

        # Calculate average losses
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_alpha = np.mean(alpha_values) if alpha_values else 0.0

        # Calculate balanced time metric
        balanced_time = 0.0
        if len(episode_states) > 0:
            states_array = np.array(episode_states)
            alphas = states_array[:, 1]  # theta_1 (pendulum angle)
            alpha_normalized = np.array([normalize_angle(a + np.pi) for a in alphas])
            upright_threshold = 0.17  # about 10 degrees

            for i in range(len(alpha_normalized)):
                if abs(alpha_normalized[i]) < upright_threshold:
                    dt_value = env.time_history[i] if env.variable_dt and i < len(env.time_history) else env.dt
                    balanced_time += dt_value

        # Get current environment parameters
        current_params = env.get_current_parameters()

        # Instead of saving the buffer to disk, put it directly in the results
        # Convert the buffer to a list of tuples for easier serialization
        buffer_data = replay_buffer.buffer[:len(replay_buffer)]  # Get all transitions

        # Prepare results
        results = {
            'worker_id': worker_id,
            'episode_num': episode_num,
            'hyperparams': hyperparams,
            'reward': episode_reward,
            'balanced_time': balanced_time,
            'critic_loss': avg_critic_loss,
            'actor_loss': avg_actor_loss,
            'alpha': avg_alpha if isinstance(avg_alpha, float) else float(avg_alpha),
            'model_path': model_path,
            'replay_buffer': buffer_data,  # Include the full buffer directly
            'episode_duration': episode_duration,
            'env_params': current_params,
            'episode_length': len(episode_states),
            'plot_path': plot_path,
            # Only store sample data for metrics, not full trajectories
            'episode_states_sample': np.array(episode_states[:100]) if len(episode_states) > 0 else None,
            'episode_actions_sample': np.array(episode_actions[:100]) if len(episode_actions) > 0 else None,
            'step_rewards_sample': step_rewards[:100] if len(step_rewards) > 0 else None
        }

        # Put results in the queue for the main process to process
        result_queue.put(results)

        if verbose:
            print(
                f"Episode {episode_num}, Worker {worker_id} completed. Reward: {episode_reward:.2f}, Balanced: {balanced_time:.2f}s, Time: {episode_duration:.1f}s")

        return True

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Episode {episode_num}, Worker {worker_id} encountered an error: {str(e)}")
        print(error_traceback)

        # Put error information in the queue
        result_queue.put({
            'worker_id': worker_id,
            'episode_num': episode_num,
            'error': str(e),
            'traceback': error_traceback
        })

        return False


class OptimizedCPUAffinityTrainer:
    """Trainer that keeps only the best episode in RAM and assigns specific CPU cores to workers."""

    def __init__(
            self,
            num_workers=3,
            max_episodes=500,
            hidden_dim=256,
            state_dim=6,
            action_dim=1,
            output_dir="episode_parallel_results",
            reserved_cores=None,
            verbose=False
    ):
        self.num_workers = num_workers
        self.max_episodes = max_episodes
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.verbose = verbose

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.output_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create subdirectories
        self.best_episodes_dir = os.path.join(self.output_dir, "best_episodes")
        os.makedirs(self.best_episodes_dir, exist_ok=True)

        # Reserve CPU cores
        self.reserved_cores = reserved_cores if reserved_cores else []

        # Get available cores
        all_cores = list(range(psutil.cpu_count(logical=True)))
        self.worker_available_cores = [c for c in all_cores if c not in self.reserved_cores]

        # Distribute cores among workers
        self.worker_core_assignments = self._distribute_cores_to_workers()

        # Tracking the best hyperparameters
        self.best_hyperparams = None
        self.best_reward = -float('inf')
        self.best_model_path = None

        # Track hyperparameter evolution
        self.hyperparameter_history = []

        # Results tracking
        self.episode_results = []
        self.best_episode_results = []

        # Base model path - will be updated as we find better models
        self.current_model_path = None

        # Current replay buffer path
        self.current_replay_buffer = None

        # Ensure reproducibility with some randomness
        self.base_seed = 42 + int(time()) % 10000

        # Training metrics
        self.start_time = None
        self.total_training_time = 0
        self.iteration_times = []

    def _distribute_cores_to_workers(self):
        """Distribute available cores among workers"""
        worker_core_assignments = []

        # Special case: if we have the same number of workers as available cores,
        # each worker gets one core
        if self.num_workers == len(self.worker_available_cores):
            return [[core] for core in self.worker_available_cores]

        # Special case: if we have fewer workers than available cores,
        # distribute cores evenly among workers
        if self.num_workers <= len(self.worker_available_cores):
            cores_per_worker = len(self.worker_available_cores) // self.num_workers
            extra_cores = len(self.worker_available_cores) % self.num_workers

            start_idx = 0
            for i in range(self.num_workers):
                # Assign an extra core to early workers if there are extras
                worker_core_count = cores_per_worker + (1 if i < extra_cores else 0)
                worker_cores = self.worker_available_cores[start_idx:start_idx + worker_core_count]
                worker_core_assignments.append(worker_cores)
                start_idx += worker_core_count

            return worker_core_assignments

        # Case: more workers than cores - each worker gets assigned to a core
        # in a round-robin fashion
        for i in range(self.num_workers):
            core_idx = i % len(self.worker_available_cores)
            worker_core_assignments.append([self.worker_available_cores[core_idx]])

        return worker_core_assignments

    def generate_hyperparameters(self, include_best=False):
        """Generate hyperparameter sets for workers with normal distribution around best values."""
        hyperparams_list = []

        # Define parameter ranges and standard deviations for variation
        param_ranges = {
            'lr': (0.000001, 0.001),  # Learning rate range
            'gamma': (0.97, 0.999),  # Discount factor range
            'tau': (0.001, 0.02),  # Soft update rate range
            'alpha': (0.02, 0.35)  # Temperature parameter range
        }

        # Define standard deviations for each parameter (as percentage of range)
        param_std_devs = {
            'lr': 0.3,  # 30% of range for learning rate
            'gamma': 0.1,  # 10% of range for gamma
            'tau': 0.2,  # 20% of range for tau
            'alpha': 0.15  # 15% of range for alpha
        }

        # Include the best hyperparameters from previous episode if available
        if include_best and self.best_hyperparams is not None:
            print(f"Reusing best hyperparameters from previous episode for Worker 0")
            best_params = deepcopy(self.best_hyperparams)

            # Ensure updates_per_step is set to 1 regardless of what was in best_params
            modified_best_params = deepcopy(best_params)
            modified_best_params['updates_per_step'] = 1

            hyperparams_list.append(modified_best_params)
        # For the first episode, include default hyperparameters for one worker
        elif not include_best:  # This means we're in the first episode
            print(f"Using default hyperparameters for Worker 0 in first episode")
            default_params = {
                'lr': 3e-4,  # Default learning rate
                'gamma': 0.99,  # Default discount factor
                'tau': 0.005,  # Default soft update rate
                'alpha': 0.2,  # Default temperature parameter
                'automatic_entropy_tuning': True,
                'updates_per_step': 1,  # Fixed to 1
                'seed': self.base_seed  # Use base seed for default params
            }
            hyperparams_list.append(default_params)
            best_params = default_params

        # Generate variations around the best hyperparameters for remaining workers
        remaining = self.num_workers - len(hyperparams_list)
        for i in range(remaining):
            params = {}

            # Generate variation for each parameter
            for param_name, range_vals in param_ranges.items():
                min_val, max_val = range_vals

                # Get the best value as the mean
                best_value = best_params[param_name]

                # Calculate the absolute standard deviation
                std_dev = param_std_devs[param_name] * (max_val - min_val)

                # Special case for learning rate - use log scale
                if param_name == 'lr':
                    # Convert to log space
                    log_min, log_max = np.log10(min_val), np.log10(max_val)
                    log_best = np.log10(best_value)
                    log_std = param_std_devs[param_name] * (log_max - log_min)

                    # Sample in log space
                    log_value = np.random.normal(log_best, log_std)

                    # Convert back and clip
                    value = 10 ** np.clip(log_value, log_min, log_max)
                else:
                    # Sample with normal distribution around best value
                    value = np.random.normal(best_value, std_dev)

                    # Clip to valid range
                    value = np.clip(value, min_val, max_val)

                params[param_name] = value

            # Handle boolean parameter separately
            # 80% chance to use the same value as best, 20% to flip
            if np.random.random() < 0.8:
                params['automatic_entropy_tuning'] = best_params['automatic_entropy_tuning']
            else:
                params['automatic_entropy_tuning'] = not best_params['automatic_entropy_tuning']

            # Always set updates_per_step to 1
            params['updates_per_step'] = 1

            # Set seed
            params['seed'] = self.base_seed + i + 1

            hyperparams_list.append(params)

        return hyperparams_list

    def generate_environment_params(self):
        """Generate environment parameters."""
        params = {
            'variable_dt': True,
            'param_variation': 0.2,
            'fixed_params': False,
            'voltage_range': (4.0, 18.0)
        }
        return params

    def run(self):
        """Run the parallel training process, keeping only the best episode in RAM."""
        self.start_time = time()

        # Track rewards
        all_rewards = []
        best_rewards = []
        all_balanced_times = []

        # Create a manager for sharing data between processes
        manager = Manager()

        for episode in range(self.max_episodes):
            episode_start_time = time()

            print(f"\n=== Starting Episode {episode + 1}/{self.max_episodes} ===")

            # Use a unique seed for this episode
            episode_seed = self.base_seed + episode * 997

            # Generate environment parameters
            env_params = self.generate_environment_params()
            print(f"Environment params: {env_params}")

            # Generate hyperparameters for workers
            hyperparams_list = self.generate_hyperparameters(include_best=(episode > 0))

            # Create a shared queue for results
            result_queue = manager.Queue()

            # Start worker processes with CPU affinity
            processes = []
            for worker_id in range(self.num_workers):
                # Get assigned cores for this worker
                worker_cores = self.worker_core_assignments[worker_id]

                # Package parameters, including the replay buffer directly
                params = {
                    'hyperparams': hyperparams_list[worker_id],
                    'env_params': env_params,
                    'episode_seed': episode_seed,
                    'base_model_path': self.current_model_path,
                    'replay_buffer': self.current_replay_buffer,  # Pass buffer directly
                    'hidden_dim': self.hidden_dim,
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'output_dir': self.output_dir
                }

                p = mp.Process(
                    target=worker_with_affinity,
                    args=(worker_id, params, episode, worker_cores, result_queue, self.verbose)
                )
                p.start()
                processes.append(p)

            # Process results as they arrive, keeping track of the best
            episode_results = []
            episode_rewards = []
            episode_balanced_times = []
            best_result = None
            best_result_reward = -float('inf')

            # Collect results as they come in
            completed_workers = 0
            while completed_workers < self.num_workers:
                # Get result from queue (blocking until a result is available)
                result = result_queue.get()

                if 'error' not in result:
                    # Valid result
                    episode_results.append(result)

                    # Track metrics
                    episode_rewards.append(result['reward'])
                    episode_balanced_times.append(result.get('balanced_time', 0))

                    # Check if this is the best result so far
                    if result['reward'] > best_result_reward:
                        best_result = result
                        best_result_reward = result['reward']

                        print(
                            f"New best for this episode: Worker {result['worker_id']} with reward {result['reward']:.2f}")
                else:
                    print(f"Worker {result['worker_id']} failed with error: {result['error']}")

                completed_workers += 1

            # Make sure all processes are finished
            for p in processes:
                p.join()

            # Calculate episode duration
            episode_duration = time() - episode_start_time
            self.iteration_times.append(episode_duration)

            # Store episode results for plotting
            all_rewards.append(episode_rewards)
            all_balanced_times.append(episode_balanced_times)

            # Process the best result (if any valid results were returned)
            if best_result:
                print(f"\nBest performer: Worker {best_result['worker_id']}")
                print(f"Episode reward: {best_result['reward']:.2f}")
                print(f"Balanced time: {best_result['balanced_time']:.2f}s")
                print(f"Hyperparameters: {best_result['hyperparams']}")

                # Print environment parameters of the best result
                env_params = best_result['env_params']
                print(f"Environment parameters: Rm={env_params['Rm']:.4f}, Km={env_params['Km']:.6f}, "
                      f"mL={env_params['mL']:.6f}, k={env_params['k']:.6f}, "
                      f"max_voltage={env_params['max_voltage']:.2f}V")

                best_rewards.append(best_result['reward'])

                # Copy the best episode plot to the best_episodes directory
                if 'plot_path' in best_result and os.path.exists(best_result['plot_path']):
                    best_plot_dest = os.path.join(self.best_episodes_dir, f"best_episode_{episode + 1}.png")
                    import shutil
                    shutil.copy2(best_result['plot_path'], best_plot_dest)

                # Update overall best if this episode's best is better than the global best
                if best_result['reward'] > self.best_reward:
                    self.best_reward = best_result['reward']
                    self.best_hyperparams = deepcopy(best_result['hyperparams'])
                    self.best_model_path = best_result['model_path']
                    print(f"New global best performance! Reward: {self.best_reward:.2f}")
                else:
                    print(f"No improvement over previous global best: {self.best_reward:.2f}")

                # Always use the best from the current episode for the next episode
                self.current_model_path = best_result['model_path']
                self.current_replay_buffer = best_result['replay_buffer']  # Store buffer directly

                # Store the best result for this episode
                self.best_episode_results.append(best_result)

                # Track hyperparameter evolution
                self.hyperparameter_history.append({
                    'episode': episode,
                    'hyperparams': best_result['hyperparams'],
                    'reward': best_result['reward'],
                    'balanced_time': best_result.get('balanced_time', 0)
                })

            # Store summarized results for this episode - only keep metrics, not full trajectories
            # This reduces memory usage compared to storing all worker results
            summarized_results = []
            for result in episode_results:
                result_copy = result.copy()
                # Remove large objects
                if 'episode_states_sample' in result_copy:
                    del result_copy['episode_states_sample']
                if 'episode_actions_sample' in result_copy:
                    del result_copy['episode_actions_sample']
                if 'step_rewards_sample' in result_copy:
                    del result_copy['step_rewards_sample']
                summarized_results.append(result_copy)

            self.episode_results.append(summarized_results)

            # Plot progress
            self._plot_training_progress(best_rewards, all_rewards, episode, all_balanced_times)

            # Plot hyperparameter evolution
            self._plot_hyperparameter_evolution(episode)

            # Calculate elapsed and estimated time
            elapsed_time = time() - self.start_time
            avg_iteration_time = np.mean(self.iteration_times)
            estimated_total_time = avg_iteration_time * self.max_episodes
            remaining_time = estimated_total_time - elapsed_time

            # Print timing information
            elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
            remaining_time_str = str(timedelta(seconds=int(remaining_time)))
            print(f"\nEpisode duration: {episode_duration:.1f}s")
            print(f"Elapsed time: {elapsed_time_str}")
            print(f"Estimated remaining time: {remaining_time_str}")
            print(f"Estimated completion: {datetime.now() + timedelta(seconds=int(remaining_time))}")

            # Save just the hyperparameter history and best results to disk periodically
            if (episode + 1) % 10 == 0:
                self.save_results(os.path.join(self.output_dir, f"results_snapshot_ep{episode + 1}.json"))

        # Training complete
        total_time = time() - self.start_time
        self.total_training_time = total_time
        print(f"\n=== Training Complete ===")
        print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
        print(f"Best reward: {self.best_reward:.2f}")
        print(f"Best hyperparameters: {self.best_hyperparams}")

        # Save final results
        self.save_results(os.path.join(self.output_dir, "results_final.json"))

        # Create final hyperparameter plots
        self._plot_final_hyperparameter_analysis()

        # Load the best model for return
        if self.best_model_path and os.path.exists(self.best_model_path):
            best_agent = SACAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim
            )
            best_agent.actor.load_state_dict(torch.load(self.best_model_path, map_location='cpu'))
            return best_agent, self.best_hyperparams, self.best_reward
        else:
            print("Warning: Best model not found. Returning None.")
            return None, self.best_hyperparams, self.best_reward

    def _plot_training_progress(self, best_rewards, all_rewards, episode, all_balanced_times=None):
        """Plot training progress across episodes."""
        try:
            # Plot best rewards over episodes
            plt.figure(figsize=(15, 12))

            # Plot 1: Best reward for each episode
            plt.subplot(3, 1, 1)
            plt.plot(range(1, len(best_rewards) + 1), best_rewards, 'b-', marker='o')
            plt.ylabel('Best Reward per Episode')
            plt.title('Training Progress')
            plt.grid(True)

            # Plot 2: Box plot of rewards for all workers
            plt.subplot(3, 1, 2)
            plt.boxplot(all_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Worker Rewards')
            plt.grid(True)

            # Plot 3: Balanced time (if available)
            if all_balanced_times and len(all_balanced_times) > 0:
                plt.subplot(3, 1, 3)
                # Extract the best balanced time from each episode
                best_balanced_times = [max(times) if times else 0 for times in all_balanced_times]
                plt.plot(range(1, len(best_balanced_times) + 1), best_balanced_times, 'g-', marker='s')
                plt.xlabel('Episode')
                plt.ylabel('Best Balanced Time (s)')
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"training_progress_ep{episode + 1}.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting training progress: {e}")

    def _plot_hyperparameter_evolution(self, episode):
        """Plot the evolution of hyperparameters."""
        if len(self.hyperparameter_history) < 2:
            return  # Need at least 2 points to plot evolution

        try:
            # Extract hyperparameter values over episodes
            episodes = [h['episode'] + 1 for h in self.hyperparameter_history]  # +1 for 1-based indexing
            rewards = [h['reward'] for h in self.hyperparameter_history]

            # Continuous hyperparameters
            lr_values = [h['hyperparams']['lr'] for h in self.hyperparameter_history]
            gamma_values = [h['hyperparams']['gamma'] for h in self.hyperparameter_history]
            tau_values = [h['hyperparams']['tau'] for h in self.hyperparameter_history]
            alpha_values = [h['hyperparams']['alpha'] for h in self.hyperparameter_history]

            # Discrete hyperparameters
            auto_entropy = [int(h['hyperparams']['automatic_entropy_tuning']) for h in self.hyperparameter_history]
            updates_per_step = [h['hyperparams']['updates_per_step'] for h in self.hyperparameter_history]

            # Create the figure
            plt.figure(figsize=(15, 15))

            # Plot rewards
            plt.subplot(4, 2, 1)
            plt.plot(episodes, rewards, 'b-', marker='o')
            plt.title('Best Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)

            # Plot continuous hyperparameters
            plt.subplot(4, 2, 2)
            plt.plot(episodes, lr_values, 'r-', marker='s')
            plt.title('Learning Rate')
            plt.xlabel('Episode')
            plt.ylabel('LR')
            plt.yscale('log')  # Use log scale for learning rate
            plt.grid(True)

            plt.subplot(4, 2, 3)
            plt.plot(episodes, gamma_values, 'g-', marker='d')
            plt.title('Discount Factor (gamma)')
            plt.xlabel('Episode')
            plt.ylabel('Gamma')
            plt.grid(True)

            plt.subplot(4, 2, 4)
            plt.plot(episodes, tau_values, 'm-', marker='*')
            plt.title('Soft Update Rate (tau)')
            plt.xlabel('Episode')
            plt.ylabel('Tau')
            plt.grid(True)

            plt.subplot(4, 2, 5)
            plt.plot(episodes, alpha_values, 'c-', marker='^')
            plt.title('Temperature Parameter (alpha)')
            plt.xlabel('Episode')
            plt.ylabel('Alpha')
            plt.grid(True)

            # Plot discrete hyperparameters
            plt.subplot(4, 2, 6)
            plt.plot(episodes, auto_entropy, 'y-', marker='o', drawstyle='steps-post')
            plt.title('Automatic Entropy Tuning')
            plt.xlabel('Episode')
            plt.ylabel('Enabled (1=Yes, 0=No)')
            plt.yticks([0, 1])
            plt.grid(True)

            # Add a correlation plot between reward and learning rate
            plt.subplot(4, 2, 7)
            plt.scatter(lr_values, rewards)
            plt.title('Reward vs. Learning Rate')
            plt.xlabel('Learning Rate')
            plt.ylabel('Reward')
            plt.xscale('log')
            plt.grid(True)

            # Add a correlation plot between reward and gamma
            plt.subplot(4, 2, 8)
            plt.scatter(gamma_values, rewards)
            plt.title('Reward vs. Gamma')
            plt.xlabel('Gamma')
            plt.ylabel('Reward')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"hyperparameter_evolution_ep{episode + 1}.png"))
            plt.close()

        except Exception as e:
            print(f"Error plotting hyperparameter evolution: {e}")

    def _plot_final_hyperparameter_analysis(self):
        """Create comprehensive plots for hyperparameter analysis at the end of training."""
        if len(self.hyperparameter_history) < 3:
            return  # Need at least 3 points for meaningful analysis

        try:
            # Extract data
            episodes = [h['episode'] + 1 for h in self.hyperparameter_history]  # +1 for 1-based indexing
            rewards = [h['reward'] for h in self.hyperparameter_history]
            balanced_times = [h.get('balanced_time', 0) for h in self.hyperparameter_history]

            # Create a figure for hyperparameter importance analysis
            plt.figure(figsize=(16, 10))

            # Hyperparameters to analyze
            param_names = ['lr', 'gamma', 'tau', 'alpha', 'automatic_entropy_tuning', 'updates_per_step']

            # Create correlation plots for each hyperparameter vs. reward
            for i, param in enumerate(param_names):
                plt.subplot(2, 3, i + 1)

                # Extract parameter values
                if param == 'automatic_entropy_tuning':
                    # Convert boolean to int for plotting
                    values = [int(h['hyperparams'][param]) for h in self.hyperparameter_history]
                    plt.yticks([0, 1], ['False', 'True'])
                else:
                    values = [h['hyperparams'][param] for h in self.hyperparameter_history]

                # Create scatter plot
                plt.scatter(rewards, values)

                # Add best fit line if the parameter is continuous
                if param not in ['automatic_entropy_tuning', 'updates_per_step']:
                    try:
                        # Only compute correlation if we have enough different values
                        unique_values = set(values)
                        if len(unique_values) > 2:
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(rewards, values)
                            line_x = np.array([min(rewards), max(rewards)])
                            line_y = slope * line_x + intercept
                            plt.plot(line_x, line_y, 'r-', alpha=0.7)
                            plt.title(f'{param} vs. Reward (r={r_value:.2f}, p={p_value:.3f})')
                        else:
                            plt.title(f'{param} vs. Reward')
                    except:
                        plt.title(f'{param} vs. Reward')
                else:
                    plt.title(f'{param} vs. Reward')

                plt.xlabel('Reward')
                plt.ylabel(param)
                plt.grid(True)

                # Use log scale for learning rate
                if param == 'lr':
                    plt.yscale('log')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "hyperparameter_importance.png"))
            plt.close()

            # Create evolution over time plots
            plt.figure(figsize=(15, 10))

            # Plot the rewards and balanced times
            plt.subplot(2, 1, 1)
            plt.plot(episodes, rewards, 'b-', marker='o', label='Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward Evolution')
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(episodes, balanced_times, 'g-', marker='s', label='Balanced Time')
            plt.xlabel('Episode')
            plt.ylabel('Balanced Time (s)')
            plt.title('Balance Performance Evolution')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "performance_evolution.png"))
            plt.close()

        except Exception as e:
            print(f"Error creating final hyperparameter analysis: {e}")

    def save_results(self, filename):
        """Save only essential training results to a JSON file."""

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

        # Extract only the essential information
        serialized_hyperparameter_history = convert_to_serializable(self.hyperparameter_history)

        # Only store summaries of the best results, not all worker results
        best_episode_summaries = []
        for result in self.best_episode_results:
            summary = {
                'episode': result['episode_num'],
                'worker_id': result['worker_id'],
                'reward': result['reward'],
                'balanced_time': result.get('balanced_time', 0),
                'hyperparams': result['hyperparams'],
                'model_path': os.path.basename(result['model_path']),
                'plot_path': os.path.basename(result['plot_path']) if 'plot_path' in result else None,
            }
            best_episode_summaries.append(summary)

        results = {
            'run_id': self.run_id,
            'best_reward': float(self.best_reward),
            'best_hyperparams': convert_to_serializable(self.best_hyperparams),
            'hidden_dim': self.hidden_dim,
            'num_workers': self.num_workers,
            'max_episodes': self.max_episodes,
            'episodes_completed': len(self.episode_results),
            'best_model_path': os.path.basename(self.best_model_path) if self.best_model_path else None,
            'total_training_time': self.total_training_time,
            'iteration_times': convert_to_serializable(self.iteration_times),
            'hyperparameter_history': serialized_hyperparameter_history,
            'best_episode_summaries': best_episode_summaries
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
