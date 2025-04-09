import multiprocessing as mp
import os
import argparse
import torch
import numpy as np
from time import time
import psutil

# Import the custom trainer from episode_parallel_trainer
from episode_parallel_trainer import OptimizedCPUAffinityTrainer


if __name__ == "__main__":

    # Get CPU information
    total_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Episode-Level Parallel Training with RAM-Only Best Results')
    parser.add_argument('--workers', type=int, default=6,
                      help=f'Number of parallel workers per episode (default: 6)')
    parser.add_argument('--reserve-cores', type=int, default=2,
                      help='Number of CPU cores to completely reserve (default: 2)')
    parser.add_argument('--reserve-specific', type=str, default=None,
                      help='Comma-separated list of specific core IDs to reserve (e.g., "0,1")')
    parser.add_argument('--episodes', type=int, default=500, help='Total number of episodes to train')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer size for all networks')
    parser.add_argument('--output-dir', type=str, default='episode_parallel_results',
                      help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output including core assignments')

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

    print("=" * 80)
    print("Starting RAM-Only Best Results Parallel Training for Inverted Pendulum Control")
    print(f"CPU: {physical_cores} physical cores, {total_cores} logical processors")
    print(f"Reserved cores: {reserved_cores}")
    print(f"Verbose mode: {'Enabled' if args.verbose else 'Disabled'}")
    print(f"Total episodes: {args.episodes}")
    print(f"Network architecture: {args.hidden_dim} units per hidden layer")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Start measuring total time
    start_time = time()

    # Initialize the optimized trainer
    trainer = OptimizedCPUAffinityTrainer(
        num_workers=args.workers,
        max_episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir,
        reserved_cores=reserved_cores,
        verbose=args.verbose
    )

    # Run the parallel training
    best_agent, best_hyperparams, best_reward = trainer.run()

    # Print total runtime
    total_time = time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "=" * 80)
    print(f"RAM-Only Episode-Level Parallel Training Complete!")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Best hyperparameters: {best_hyperparams}")
    print("=" * 80)