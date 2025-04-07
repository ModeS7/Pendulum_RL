import multiprocessing as mp
import os
import argparse
import torch
import numpy as np
from time import time


from episode_parallel_trainer import SimpleParallelTrainer


if __name__ == "__main__":
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Episode-Level Parallel Training for Inverted Pendulum Control')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers per episode')
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
    # 'spawn' is more reliable for PyTorch across platforms
    mp.set_start_method('spawn')

    print("=" * 80)
    print("Starting Episode-Level Parallel Training for Inverted Pendulum Control")
    print(f"Workers per episode: {args.workers}")
    print(f"Total episodes: {args.episodes}")
    print(f"Network architecture: {args.hidden_dim} units per hidden layer")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Start measuring total time
    start_time = time()

    # Initialize and run the simplified parallel trainer
    trainer = SimpleParallelTrainer(
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