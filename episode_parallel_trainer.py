import multiprocessing as mp
import numpy as np
import torch
import os
import json
import pickle
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import tempfile
from datetime import datetime, timedelta

# Import your pendulum code
from SimRL import PendulumEnv, SACAgent, ReplayBuffer, plot_training_episode, normalize_angle


def worker_training_job(worker_id, param_file, result_file, episode_num):
    """
    This function will be run in a separate process.
    It doesn't receive any PyTorch tensors directly.

    Args:
        worker_id: ID of this worker
        param_file: File path containing parameters
        result_file: File path to write results
        episode_num: Current episode number
    """
    try:
        # Load parameters from file
        with open(param_file, 'rb') as f:
            params = pickle.load(f)

        hyperparams = params['hyperparams']
        env_params = params['env_params']
        base_model_path = params['base_model_path']
        replay_buffer_path = params.get('replay_buffer_path')
        hidden_dim = params['hidden_dim']
        state_dim = params['state_dim']
        action_dim = params['action_dim']

        print(f"Episode {episode_num}, Worker {worker_id} starting with hyperparams: {hyperparams}")

        # Set the seed for reproducibility
        seed = hyperparams.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create environment
        env = PendulumEnv(**env_params)

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

        # Reconstruct replay buffer if available
        replay_buffer = ReplayBuffer(100000)
        if replay_buffer_path and os.path.exists(replay_buffer_path):
            try:
                with open(replay_buffer_path, 'rb') as f:
                    replay_data = pickle.load(f)
                for transition in replay_data:
                    replay_buffer.push(*transition)
            except Exception as e:
                print(f"Error loading replay buffer: {e}")

        # Train for one episode
        batch_size = 256
        updates_per_step = hyperparams.get('updates_per_step', 1)
        output_dir = f"episode_parallel_results/episode_{episode_num}/worker_{worker_id}"
        os.makedirs(output_dir, exist_ok=True)

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

        # Plot the episode results
        plot_path = os.path.join(output_dir, "episode_plot.png")
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

        # Save model and sample of replay buffer
        model_path = os.path.join(output_dir, "actor_model.pth")
        torch.save(agent.actor.state_dict(), model_path)

        # Sample a subset of replay buffer to save
        replay_sample = []
        if len(replay_buffer) > 0:
            indices = np.random.choice(len(replay_buffer), min(5000, len(replay_buffer)), replace=False)
            for i in indices:
                replay_sample.append(replay_buffer.buffer[i])

        replay_path = os.path.join(output_dir, "replay_buffer.pkl")
        with open(replay_path, 'wb') as f:
            pickle.dump(replay_sample, f)

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
            'replay_buffer_path': replay_path,
            'episode_duration': episode_duration,
            'env_params': current_params,
            'episode_length': len(episode_states),
            'plot_path': plot_path,
            'episode_states': np.array(episode_states) if len(episode_states) < 500 else None,
            # Only store if not too large
            'episode_actions': np.array(episode_actions) if len(episode_actions) < 500 else None,
            'step_rewards': step_rewards if len(step_rewards) < 500 else None
        }

        # Save results to file
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)

        print(
            f"Episode {episode_num}, Worker {worker_id} completed. Reward: {episode_reward:.2f}, Balanced: {balanced_time:.2f}s, Time: {episode_duration:.1f}s")
        return True

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Episode {episode_num}, Worker {worker_id} encountered an error: {str(e)}")
        print(error_traceback)

        # Save error information to result file
        error_results = {
            'worker_id': worker_id,
            'episode_num': episode_num,
            'error': str(e),
            'traceback': error_traceback
        }

        with open(result_file, 'wb') as f:
            pickle.dump(error_results, f)

        return False


class SimpleParallelTrainer:
    """A simplified parallel trainer that avoids passing tensors between processes."""

    def __init__(
            self,
            num_workers=3,
            max_episodes=500,
            hidden_dim=256,
            state_dim=6,
            action_dim=1,
            output_dir="episode_parallel_results"
    ):
        self.num_workers = num_workers
        self.max_episodes = max_episodes
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.output_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create subdirectories
        self.temp_dir = os.path.join(self.output_dir, "temp")
        self.best_episodes_dir = os.path.join(self.output_dir, "best_episodes")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.best_episodes_dir, exist_ok=True)

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
        self.current_replay_path = None

        # Ensure reproducibility
        self.base_seed = 42

        # Training metrics
        self.start_time = None
        self.total_training_time = 0
        self.iteration_times = []

    def generate_hyperparameters(self, include_best=False):
        """Generate hyperparameter sets for workers."""
        hyperparams_list = []

        # Include the best hyperparameters from previous episode if available
        if include_best and self.best_hyperparams is not None:
            hyperparams_list.append(deepcopy(self.best_hyperparams))

        # Generate random hyperparameters for remaining workers
        remaining = self.num_workers - len(hyperparams_list)
        for i in range(remaining):
            params = {
                'lr': 10 ** np.random.uniform(-4, -2.3),
                'gamma': np.random.uniform(0.97, 0.995),
                'tau': np.random.uniform(0.001, 0.01),
                'automatic_entropy_tuning': bool(np.random.choice([True, False], p=[0.8, 0.2])),
                'alpha': np.random.uniform(0.05, 0.3),
                'updates_per_step': np.random.choice([1, 2, 3]),
                'seed': self.base_seed + i + 1
            }
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
        """Run the parallel training process."""
        self.start_time = time()

        # Track rewards
        all_rewards = []
        best_rewards = []
        all_balanced_times = []

        for episode in range(self.max_episodes):
            episode_start_time = time()

            print(f"\n=== Starting Episode {episode + 1}/{self.max_episodes} ===")

            # Generate environment parameters
            env_params = self.generate_environment_params()
            print(f"Environment params: {env_params}")

            # Generate hyperparameters for workers
            hyperparams_list = self.generate_hyperparameters(include_best=(episode > 0))

            # Create parameter files and result files for each worker
            param_files = []
            result_files = []
            for worker_id in range(self.num_workers):
                param_file = os.path.join(self.temp_dir, f"params_ep{episode}_worker{worker_id}.pkl")
                result_file = os.path.join(self.temp_dir, f"results_ep{episode}_worker{worker_id}.pkl")

                # Package parameters
                params = {
                    'hyperparams': hyperparams_list[worker_id],
                    'env_params': env_params,
                    'base_model_path': self.current_model_path,
                    'replay_buffer_path': self.current_replay_path,
                    'hidden_dim': self.hidden_dim,
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim
                }

                # Save parameters to file
                with open(param_file, 'wb') as f:
                    pickle.dump(params, f)

                param_files.append(param_file)
                result_files.append(result_file)

            # Start worker processes
            processes = []
            for worker_id in range(self.num_workers):
                p = mp.Process(
                    target=worker_training_job,
                    args=(worker_id, param_files[worker_id], result_files[worker_id], episode)
                )
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Collect results
            episode_results = []
            for result_file in result_files:
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'rb') as f:
                            result = pickle.load(f)
                        if 'error' not in result:
                            episode_results.append(result)
                        else:
                            print(f"Worker {result['worker_id']} failed with error: {result['error']}")
                    except Exception as e:
                        print(f"Error loading result file {result_file}: {e}")

            # Calculate episode duration
            episode_duration = time() - episode_start_time
            self.iteration_times.append(episode_duration)

            # Calculate elapsed and estimated time
            elapsed_time = time() - self.start_time
            avg_iteration_time = np.mean(self.iteration_times)
            estimated_total_time = avg_iteration_time * self.max_episodes
            remaining_time = estimated_total_time - elapsed_time

            # Find the best performer
            if episode_results:
                # Sort by episode reward
                episode_results.sort(key=lambda x: x['reward'], reverse=True)
                best_result = episode_results[0]

                print(f"\nBest performer: Worker {best_result['worker_id']}")
                print(f"Episode reward: {best_result['reward']:.2f}")
                print(f"Balanced time: {best_result['balanced_time']:.2f}s")
                print(f"Hyperparameters: {best_result['hyperparams']}")

                # Print environment parameters of the best result
                env_params = best_result['env_params']
                print(f"Environment parameters: Rm={env_params['Rm']:.4f}, Km={env_params['Km']:.6f}, "
                      f"mL={env_params['mL']:.6f}, k={env_params['k']:.6f}, "
                      f"max_voltage={env_params['max_voltage']:.2f}V")

                # Track rewards and balanced times
                episode_rewards = [r['reward'] for r in episode_results]
                episode_balanced_times = [r.get('balanced_time', 0) for r in episode_results]
                all_rewards.append(episode_rewards)
                all_balanced_times.append(episode_balanced_times)
                best_rewards.append(best_result['reward'])

                # Update overall best if this is better
                if best_result['reward'] > self.best_reward:
                    self.best_reward = best_result['reward']
                    self.best_hyperparams = deepcopy(best_result['hyperparams'])
                    self.best_model_path = best_result['model_path']

                    # Update current model path for next episode
                    self.current_model_path = best_result['model_path']
                    self.current_replay_path = best_result['replay_buffer_path']

                    # Copy the best episode plot to the best_episodes directory
                    if 'plot_path' in best_result and os.path.exists(best_result['plot_path']):
                        best_plot_dest = os.path.join(self.best_episodes_dir, f"best_episode_{episode + 1}.png")
                        import shutil
                        shutil.copy2(best_result['plot_path'], best_plot_dest)

                    print(f"New best performance found! Reward: {self.best_reward:.2f}")
                else:
                    print(f"No improvement over previous best: {self.best_reward:.2f}")

                # Store the best result for this episode
                self.best_episode_results.append(best_result)

                # Track hyperparameter evolution
                self.hyperparameter_history.append({
                    'episode': episode,
                    'hyperparams': best_result['hyperparams'],
                    'reward': best_result['reward'],
                    'balanced_time': best_result.get('balanced_time', 0)
                })

            # Store all results for this episode
            self.episode_results.append(episode_results)

            # Plot progress
            self._plot_training_progress(best_rewards, all_rewards, episode, all_balanced_times)

            # Plot hyperparameter evolution
            self._plot_hyperparameter_evolution(episode)

            # Save intermediate state
            self.save_results(os.path.join(self.output_dir, f"results_ep{episode + 1}.json"))

            # Print timing information
            elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
            remaining_time_str = str(timedelta(seconds=int(remaining_time)))
            print(f"\nEpisode duration: {episode_duration:.1f}s")
            print(f"Elapsed time: {elapsed_time_str}")
            print(f"Estimated remaining time: {remaining_time_str}")
            print(f"Estimated completion: {datetime.now() + timedelta(seconds=int(remaining_time))}")

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
            self._final_evaluation(best_agent)
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
            plt.subplot(5, 2, 1)
            plt.plot(episodes, rewards, 'b-', marker='o')
            plt.title('Best Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)

            # Plot continuous hyperparameters
            plt.subplot(5, 2, 3)
            plt.plot(episodes, lr_values, 'r-', marker='s')
            plt.title('Learning Rate')
            plt.xlabel('Episode')
            plt.ylabel('LR')
            plt.yscale('log')  # Use log scale for learning rate
            plt.grid(True)

            plt.subplot(5, 2, 4)
            plt.plot(episodes, gamma_values, 'g-', marker='d')
            plt.title('Discount Factor (gamma)')
            plt.xlabel('Episode')
            plt.ylabel('Gamma')
            plt.grid(True)

            plt.subplot(5, 2, 5)
            plt.plot(episodes, tau_values, 'm-', marker='*')
            plt.title('Soft Update Rate (tau)')
            plt.xlabel('Episode')
            plt.ylabel('Tau')
            plt.grid(True)

            plt.subplot(5, 2, 6)
            plt.plot(episodes, alpha_values, 'c-', marker='^')
            plt.title('Temperature Parameter (alpha)')
            plt.xlabel('Episode')
            plt.ylabel('Alpha')
            plt.grid(True)

            # Plot discrete hyperparameters
            plt.subplot(5, 2, 7)
            plt.plot(episodes, auto_entropy, 'y-', marker='o', drawstyle='steps-post')
            plt.title('Automatic Entropy Tuning')
            plt.xlabel('Episode')
            plt.ylabel('Enabled (1=Yes, 0=No)')
            plt.yticks([0, 1])
            plt.grid(True)

            plt.subplot(5, 2, 8)
            plt.plot(episodes, updates_per_step, 'k-', marker='p', drawstyle='steps-post')
            plt.title('Updates Per Step')
            plt.xlabel('Episode')
            plt.ylabel('Count')
            plt.yticks([1, 2, 3])
            plt.grid(True)

            # Add a correlation plot between reward and learning rate
            plt.subplot(5, 2, 9)
            plt.scatter(lr_values, rewards)
            plt.title('Reward vs. Learning Rate')
            plt.xlabel('Learning Rate')
            plt.ylabel('Reward')
            plt.xscale('log')
            plt.grid(True)

            # Add a correlation plot between reward and gamma
            plt.subplot(5, 2, 10)
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

        # Store results for comparison
        scenario_results = {}

        for scenario in eval_scenarios:
            scenario_name = scenario.pop("name")
            scenario_dir = os.path.join(eval_dir, scenario_name)
            os.makedirs(scenario_dir, exist_ok=True)

            print(f"\n--- {scenario_name.replace('_', ' ').title()} Evaluation ---")

            rewards, times = self._evaluate_scenario(agent, scenario, scenario_dir)
            scenario_results[scenario_name] = {
                'rewards': rewards,
                'balance_times': times,
                'avg_reward': np.mean(rewards),
                'avg_balance_time': np.mean(times) if times else 0
            }

        # Create comparison plot of different scenarios
        self._plot_scenario_comparison(scenario_results, os.path.join(eval_dir, "scenario_comparison.png"))

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

        return rewards, balance_times

    def _plot_scenario_comparison(self, scenario_results, save_path):
        """Create a comparison plot of different evaluation scenarios."""
        if not scenario_results:
            return

        try:
            # Create figure
            plt.figure(figsize=(15, 10))

            # Get scenario names and sort them in a sensible order
            scenario_names = list(scenario_results.keys())
            if 'standard' in scenario_names:
                # Put standard first, then sort others by name
                scenario_names.remove('standard')
                scenario_names.sort()
                scenario_names = ['standard'] + scenario_names
            else:
                scenario_names.sort()

            # Plot average rewards
            plt.subplot(2, 1, 1)
            avg_rewards = [scenario_results[name]['avg_reward'] for name in scenario_names]
            std_rewards = [np.std(scenario_results[name]['rewards']) for name in scenario_names]

            x = np.arange(len(scenario_names))
            plt.bar(x, avg_rewards, yerr=std_rewards, capsize=8)
            plt.xticks(x, [name.replace('_', ' ').title() for name in scenario_names])
            plt.ylabel('Average Reward')
            plt.title('Performance Across Scenarios - Reward')
            plt.grid(axis='y')

            # Plot average balance times
            plt.subplot(2, 1, 2)
            avg_times = [scenario_results[name]['avg_balance_time'] for name in scenario_names]
            std_times = [
                np.std(scenario_results[name]['balance_times']) if scenario_results[name]['balance_times'] else 0
                for name in scenario_names]

            plt.bar(x, avg_times, yerr=std_times, capsize=8)
            plt.xticks(x, [name.replace('_', ' ').title() for name in scenario_names])
            plt.ylabel('Average Balance Time (s)')
            plt.title('Performance Across Scenarios - Balance Time')
            plt.grid(axis='y')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        except Exception as e:
            print(f"Error creating scenario comparison plot: {e}")

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

        # Extract relevant results, excluding paths to files and large arrays
        simplified_episode_results = []
        for episode_data in self.episode_results:
            simplified_data = []
            for worker_data in episode_data:
                worker_data_copy = worker_data.copy()
                # Remove file paths and large arrays
                if 'model_path' in worker_data_copy:
                    worker_data_copy['model_path'] = os.path.basename(worker_data_copy['model_path'])
                if 'replay_buffer_path' in worker_data_copy:
                    worker_data_copy['replay_buffer_path'] = os.path.basename(worker_data_copy['replay_buffer_path'])
                if 'plot_path' in worker_data_copy:
                    worker_data_copy['plot_path'] = os.path.basename(worker_data_copy['plot_path'])
                if 'episode_states' in worker_data_copy:
                    worker_data_copy.pop('episode_states')
                if 'episode_actions' in worker_data_copy:
                    worker_data_copy.pop('episode_actions')
                if 'step_rewards' in worker_data_copy:
                    worker_data_copy.pop('step_rewards')
                simplified_data.append(worker_data_copy)
            simplified_episode_results.append(simplified_data)

        # Prepare hyperparameter history for serialization
        serialized_hyperparameter_history = convert_to_serializable(self.hyperparameter_history)

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
            'episode_results': convert_to_serializable(simplified_episode_results)
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)