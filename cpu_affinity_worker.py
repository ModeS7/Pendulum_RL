import multiprocessing as mp
import pickle
import numpy as np
import torch
import os
import psutil
from time import time

# Import the original worker function
from episode_parallel_trainer import worker_training_job


def affinity_worker_training_job(worker_id, param_file, result_file, episode_num, cpu_cores, verbose=False):
    """Worker function that sets CPU affinity before running the original job"""
    try:
        # Set CPU affinity for this process
        proc = psutil.Process()
        if verbose:
            print(f"Worker {worker_id} setting CPU affinity to cores: {cpu_cores}")
        proc.cpu_affinity(cpu_cores)

        # Call the original worker function
        return worker_training_job(worker_id, param_file, result_file, episode_num)
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