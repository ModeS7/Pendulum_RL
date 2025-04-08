import multiprocessing as mp
import os
import argparse
import torch
import numpy as np
from time import time
import psutil

# Import the original trainer
from episode_parallel_trainer import SimpleParallelTrainer
# Import our custom worker function
from cpu_affinity_worker import affinity_worker_training_job

if __name__ == "__main__":
    # Get CPU information
    total_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Episode-Level Parallel Training for Inverted Pendulum Control')
    parser.add_argument('--workers', type=int, default=6,
                        help=f'Number of parallel workers per episode (default: 6)')
    parser.add_argument('--reserve-cores', type=int, default=1,
                        help='Number of CPU cores to completely reserve (default: 1)')
    parser.add_argument('--reserve-specific', type=str, default=None,
                        help='Comma-separated list of specific core IDs to reserve (e.g., "0,1")')
    parser.add_argument('--episodes', type=int, default=500, help='Total number of episodes to train')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer size for all networks')
    parser.add_argument('--output-dir', type=str, default='episode_parallel_results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up multiprocessing start method
    mp.set_start_method('spawn')

    # Determine which cores to reserve and which to use for workers
    all_cores = list(range(total_cores))
    reserved_cores = []

    if args.reserve_specific:
        # Reserve specific cores by ID
        try:
            reserved_cores = [int(core_id) for core_id in args.reserve_specific.split(',')]
            # Validate that specified cores exist
            for core in reserved_cores[:]:  # Use a copy for iteration
                if core < 0 or core >= total_cores:
                    print(f"Warning: Specified core {core} is out of range (0-{total_cores - 1})")
                    reserved_cores.remove(core)
        except:
            print(f"Warning: Invalid format for --reserve-specific. Using default.")
            reserved_cores = all_cores[:args.reserve_cores]
    else:
        # Reserve the specified number of cores (from the beginning)
        reserved_cores = all_cores[:args.reserve_cores]

    # The cores available for workers
    worker_available_cores = [c for c in all_cores if c not in reserved_cores]


    # Create a simple class that extends SimpleParallelTrainer to use our custom worker
    class CPUAffinityTrainer(SimpleParallelTrainer):
        def run(self):
            """Modified run method that assigns CPU cores to workers"""
            self.start_time = time()

            # Track rewards
            all_rewards = []
            best_rewards = []
            all_balanced_times = []

            for episode in range(self.max_episodes):
                episode_start_time = time()

                print(f"\n=== Starting Episode {episode + 1}/{self.max_episodes} ===")

                # Use a unique seed for this episode
                episode_seed = self.base_seed + episode * 997  # 997 is a prime number

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
                        'episode_seed': episode_seed,  # Pass the episode seed
                        'base_model_path': self.current_model_path,
                        'replay_buffer_path': self.current_replay_path,
                        'hidden_dim': self.hidden_dim,
                        'state_dim': self.state_dim,
                        'action_dim': self.action_dim
                    }

                    # Save parameters to file
                    with open(param_file, 'wb') as f:
                        import pickle
                        pickle.dump(params, f)

                    param_files.append(param_file)
                    result_files.append(result_file)

                # Distribute cores among workers
                worker_core_assignments = []
                if args.workers <= len(worker_available_cores):
                    # If we have enough cores, each worker gets at least one core
                    cores_per_worker = len(worker_available_cores) // args.workers
                    extra_cores = len(worker_available_cores) % args.workers

                    start_idx = 0
                    for i in range(self.num_workers):
                        # Give extra cores to the first few workers if we have extras
                        core_count = cores_per_worker + (1 if i < extra_cores else 0)
                        worker_cores = worker_available_cores[start_idx:start_idx + core_count]
                        worker_core_assignments.append(worker_cores)
                        start_idx += core_count
                else:
                    # More workers than cores - assign in round-robin fashion
                    for i in range(self.num_workers):
                        core_idx = i % len(worker_available_cores)
                        worker_core_assignments.append([worker_available_cores[core_idx]])

                # Start worker processes with CPU affinity
                processes = []
                for worker_id in range(self.num_workers):
                    # Get assigned cores for this worker
                    worker_cores = worker_core_assignments[worker_id]

                    p = mp.Process(
                        target=affinity_worker_training_job,
                        args=(worker_id, param_files[worker_id], result_files[worker_id], episode, worker_cores)
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

                # Rest of the method is unchanged from the original SimpleParallelTrainer
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
                import datetime
                elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
                remaining_time_str = str(datetime.timedelta(seconds=int(remaining_time)))
                print(f"\nEpisode duration: {episode_duration:.1f}s")
                print(f"Elapsed time: {elapsed_time_str}")
                print(f"Estimated remaining time: {remaining_time_str}")
                print(
                    f"Estimated completion: {datetime.datetime.now() + datetime.timedelta(seconds=int(remaining_time))}")

            # Call the other methods from the original class
            # Training complete
            total_time = time() - self.start_time
            self.total_training_time = total_time
            print(f"\n=== Training Complete ===")
            print(f"Total time: {str(datetime.timedelta(seconds=int(total_time)))}")
            print(f"Best reward: {self.best_reward:.2f}")
            print(f"Best hyperparameters: {self.best_hyperparams}")

            # Save final results
            self.save_results(os.path.join(self.output_dir, "results_final.json"))

            # Create final hyperparameter plots
            self._plot_final_hyperparameter_analysis()

            # Load the best model for return
            if self.best_model_path and os.path.exists(self.best_model_path):
                from SimRL import SACAgent
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


    print("=" * 80)
    print("Starting Episode-Level Parallel Training for Inverted Pendulum Control")
    print(f"CPU: {physical_cores} physical cores, {total_cores} logical processors")
    print(f"Reserved cores: {reserved_cores}")
    print(f"Available cores for workers: {worker_available_cores}")
    print(f"Workers per episode: {args.workers}")
    print(f"Total episodes: {args.episodes}")
    print(f"Network architecture: {args.hidden_dim} units per hidden layer")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Start measuring total time
    start_time = time()

    # Initialize and run the custom parallel trainer with CPU affinity
    trainer = CPUAffinityTrainer(
        num_workers=args.workers,
        max_episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir
    )

    # Run the parallel training
    best_agent, best_hyperparams, best_reward = trainer.run()

    # Print total runtime
    total_time = time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "=" * 80)
    print(f"Episode-Level Parallel Training Complete!")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Best hyperparameters: {best_hyperparams}")
    print("=" * 80)