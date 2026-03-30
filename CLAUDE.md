# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement Learning control of a QUBE-Servo 2 inverted pendulum, comparing SAC (Soft Actor-Critic) agents against traditional PID/LQR controllers. The project spans simulation-based training, robustness evaluation under parameter variation, and deployment to real hardware.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train SAC agent (main entry point)
python SimRL.py

# Train PPO variant
python SimRLPPO.py

# Parallel episode training
python episode_parallel_trainer.py

# Visualize reward function (interactive Tkinter GUI)
python RewardViz.py

# Run on real QUBE-Servo 2 hardware
cd QUBE_PYTHON && python main.py

# Compare PID vs RL performance
python PIDvsRL/plots.py
```

No formal test suite, linter, or build system exists.

## Architecture

**Simulation & Training (`SimRL.py`)** — self-contained ~900-line file containing:
- Lagrangian physics model with RK4 integration (`dynamics_step()`, numba-JIT compiled)
- `ParameterManager` — randomizes system parameters (mass, inertia, damping, motor constants) between episodes for robustness
- `VariableTimeGenerator` — produces non-uniform time steps to train robust policies
- `PendulumEnv` — Gym-like environment. State: `[theta_0, theta_1, theta_0_dot, theta_1_dot]` (arm angle, pendulum angle from upright, angular velocities). Observation: 6D (sin/cos of angles + scaled velocities). Action: continuous `[-1, 1]` mapped to motor voltage.
- `Actor` / `Critic` / `SACAgent` — standard SAC with dual Q-networks, automatic entropy tuning, and replay buffer
- `train()` — main loop with multi-seed aggregation, TensorBoard logging, and robustness evaluation at 15%/25%/50% parameter variation

**Hardware Control (`QUBE_PYTHON/`)** — interfaces with physical QUBE-Servo 2 over serial:
- `ControlRL.py` (and 2.0/3.0 versions) — loads trained actor network, applies to hardware readings
- `ControlPID.py`, `ControlLQR.py` — baseline controllers
- `pendulum_kalman_filter.py` — state estimation for noisy sensor data
- `main.py` — entry point with multithreading for real-time control

**Simulation Variants** — experimental iterations on the core SimRL approach:
- `SimWC.py`/`SimWCF.py` — world coordinate frame
- `SimSI.py`/`SimSI2.0.py` — SI unit variants
- `SimEI.py` — energy-informed reward
- `SimRLPar.py` — parallel training variant

**Analysis (`PIDvsRL/`, `SAC/`)** — performance comparison data, trained models, and plotting scripts.

## Key Physics & Domain Details

- Motor deadzone: voltages in `[-0.2, 0.2]V` produce zero torque
- Arm angle hard limits: `[-2.2, 2.2]` radians (reflects physical stops)
- Base max voltage: 6.0V; training varies voltage range (typically 2.0–6.0V)
- Parameter variation during training: typically 40%, tested at 15%/25%/50%
- Reward function: multi-component (upright bonus, stability, arm centering, energy efficiency, limit penalties, voltage change penalties)

## Additional Hardware Dependencies

For `QUBE_PYTHON/`: `pyserial`, `PyQt5`, `pyqtgraph` (not in requirements.txt).
