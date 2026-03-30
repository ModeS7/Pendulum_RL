# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning control system for the Quanser QUBE-Servo 2 inverted pendulum. Implements SAC/PPO training in physics-based simulation, hardware deployment via serial, and benchmarking against PID/LQR baselines.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Training
python SimRL.py                      # Main SAC training
python SimRLSimple.py                # Simplified SAC variant
python SimRLPPO.py                   # PPO training
python SimRLPar.py                   # Parallel multi-core training launcher
python episode_parallel_trainer.py   # Multi-core training with hyperparameter evolution

# Hardware deployment (requires QUBE connected via serial)
cd QUBE_PYTHON && python main.py     # Real-time control loop
cd QUBE_PYTHON && python gui.py      # PyQt GUI for monitoring

# Visualization & analysis
python RewardViz.py                  # Interactive reward function explorer
python PIDvsRL/plots.py              # PID vs RL comparison plots

# System identification
python SimSI.py                      # Basic system identification
python SimSI2.0.py                   # Enhanced system identification
python SimEI.py                      # Energy-based system identification
```

No build system, test runner, or linter is configured. No formal test suite exists.

## Architecture

**Two main pipelines: simulation training and hardware deployment.**

### Simulation & Training (`SimRL.py` and variants)

Each `SimXX.py` file is self-contained with its own copies of:
- `PendulumEnv` — Gymnasium-compatible environment with RK4 physics integration, domain randomization (`ParameterManager`), and variable timesteps (`VariableTimeGenerator`)
- `SACAgent` / `PPOAgent` — Actor-Critic networks with `ReplayBuffer`
- Training loop with TensorBoard logging

State space: `[θ_arm, θ_pendulum, θ̇_arm, θ̇_pendulum]`. Physics uses Lagrangian mechanics with motor back-EMF and viscous damping.

`SimWC.py` / `SimWCF.py` add cable dynamics and frequency-dependent damping respectively.

### Hardware Control (`QUBE_PYTHON/`)

- `main.py` — Entry point; runs the control loop calling `control_system()` from `control.py`
- `QUBE.py` — Serial protocol handler for QUBE hardware
- `ControlRL.py` / `ControlPID.py` / `ControlLQR.py` — Swappable controller implementations
- `gui.py` + `liveplot.py` — PyQt5 real-time visualization
- `com.py` — COM port and serial settings
- `config.py` — Plot display settings

### Parallel Training (`episode_parallel_trainer.py`, `SimRLPar.py`)

`OptimizedCPUAffinityTrainer` runs per-episode training across cores with Gaussian hyperparameter evolution, selecting best configs by reward.

### Supporting Artifacts

- `models/` — Trained PyTorch policy weights (`.pth` files)
- `matlab/` — MATLAB validation: parameter estimation (WyNDA), model comparison, dynamics simulation
- `PIDvsRL/` — Benchmarking data (Excel/CSV) comparing PID and RL at various voltage limits
- `QUBE/examples/` — Arduino firmware for encoder reading and serial communication

## Key Dependencies

PyTorch (torch, torchrl), Gymnasium, NumPy, Numba (JIT for physics), SciPy, PySerial, PyQt5, pyqtgraph, TensorBoard, lion-pytorch, psutil.

## Important Notes

- System parameters (motor resistance, damping coefficients, etc.) are hardcoded in each `SimXX.py` file — they are not shared via a central config.
- The `SimXX.py` variants duplicate significant code (env, agent, buffer). Changes to core logic may need replication across files.
- Hardware code assumes a specific COM port configured in `QUBE_PYTHON/com.py`.
