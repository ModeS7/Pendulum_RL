# Pendulum_RL

Reinforcement learning controller for the Quanser QUBE-Servo 2 inverted pendulum. This project trains a Soft Actor-Critic (SAC) agent in simulation and deploys it on real hardware, comparing performance against classical PID and LQR controllers.

## Overview

The project covers the full pipeline from system identification and simulation to RL training and hardware deployment:

1. **System Identification** — Estimate physical parameters (damping, inertia, motor constants) from real pendulum data using adaptive observers and Kalman filters.
2. **Simulation** — Physics-based simulator with RK4 integration, variable timestep support, and domain randomization over physical parameters for robust policy training.
3. **RL Training** — SAC agent with Gaussian policy, twin critics, and entropy regularization. Supports parallel episode training with hyperparameter evolution across CPU cores.
4. **Hardware Deployment** — Real-time control loop communicating with the QUBE-Servo via serial. Includes a PyQt GUI for live monitoring and parameter tuning.
5. **Benchmarking** — Side-by-side comparison of RL, PID, and LQR controllers at various voltage limits on both simulated and real hardware.

## Project Structure

```
Pendulum_RL/
├── SimRL.py                    # Main SAC training environment (simulation)
├── SimRLPPO.py                 # PPO-based training variant
├── SimRLSimple.py              # Simplified SAC training loop
├── SimRLTemp.py                # SAC with adaptive temperature scaling
├── SimRLPar.py                 # Parallel training launcher
├── episode_parallel_trainer.py # Multi-core episode training with hyperparameter evolution
├── cpu_affinity_worker.py      # CPU affinity management for parallel workers
├── SimSI.py                    # System identification (simplified 2DOF model)
├── SimSI2.0.py                 # System identification v2
├── SimEI.py                    # Energy-based system identification with cable effects
├── SimWC.py                    # Simulation with cable oscillation modeling
├── SimWCF.py                   # Simulation with frequency-dependent damping
├── RewardViz.py                # Reward and training metric visualization
├── requirements.txt            # Python dependencies
│
├── QUBE_PYTHON/                # Hardware interface and real-time control
│   ├── main.py                 # Entry point — initializes QUBE and control loop
│   ├── QUBE.py                 # Hardware abstraction (serial communication)
│   ├── com.py                  # Serial port configuration
│   ├── config.py               # Plotting and system configuration
│   ├── control.py              # Control interface (user-defined control law)
│   ├── ControlPID.py           # PID cascade controller
│   ├── ControlLQR.py           # LQR state-feedback controller
│   ├── ControlRL.py            # RL controller (deployed policy)
│   ├── ControlRL2.0.py         # Enhanced RL controller with improved architecture
│   ├── ControlRL3.0.py         # Refined RL controller with normalized state spaces
│   ├── PID.py                  # PID controller class
│   ├── pendulum_kalman_filter.py # Extended Kalman filter for state estimation
│   ├── inverted_pendulum.py    # Pendulum physics model
│   ├── gui.py                  # PyQt GUI for live monitoring
│   ├── liveplot.py             # Real-time plotting utilities
│   ├── logger.py               # Experiment data logging
│   ├── install.py              # Dependency installer
│   ├── TestQube.py             # Hardware and control validation tests
│   └── Readme.txt              # Quick-start guide for QUBE setup
│
├── PIDvsRL/                    # Controller comparison data and plots
│   ├── plots.py                # Generates PID vs RL comparison figures
│   ├── PID/                    # PID experiment data (.xlsx) at various voltage limits
│   └── RL/                     # RL experiment data (.csv) at various voltage limits
│       └── Filter/             # Filtered RL data
│
├── QUBE/                       # Arduino firmware for QUBE-Servo
│   └── examples/
│       ├── Inverted_Pendulum_data/  # Data collection firmware and analysis
│       └── Python_Serial/           # Serial communication firmware
│
├── validated models/           # Trained model weights (.pth)
│
├── sim_constant_voltage.m      # MATLAB: Simple pendulum dynamics with constant input
├── sim_time_varying_voltage.m  # MATLAB: Pendulum dynamics with time-varying input
├── wynda_parameter_estimation.m # MATLAB: WyNDA parameter estimation (AO + AKF)
└── learned_vs_physics_model.m  # MATLAB: Data-driven vs physics model validation
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- Quanser QUBE-Servo 2 (for hardware deployment)
- Arduino IDE (for flashing QUBE firmware)

### Installation

```bash
git clone https://github.com/ModeS7/Pendulum_RL.git
cd Pendulum_RL
pip install -r requirements.txt
```

### Training in Simulation

Run the main SAC training loop:

```bash
python SimRL.py
```

For parallel training across multiple CPU cores:

```bash
python SimRLPar.py
```

### Hardware Deployment

1. Flash the Arduino firmware from `QUBE/examples/Inverted_Pendulum_data/Inverted_Pendulum_data.ino` to the QUBE-Servo.
2. Set the correct COM port in `QUBE_PYTHON/com.py`.
3. Install hardware dependencies:
   ```bash
   pip install pyserial PyQt5 pyqtgraph
   ```
4. Run the control loop:
   ```bash
   cd QUBE_PYTHON
   python main.py
   ```

See `QUBE_PYTHON/Readme.txt` for a detailed hardware setup guide.

## System Model

The QUBE-Servo is modeled as a 2-DOF system with states `[θ_arm, θ_pendulum, θ̇_arm, θ̇_pendulum]`. The simulation uses 4th-order Runge-Kutta integration with configurable timesteps and includes:

- Motor resistance, back-EMF, and voltage dead zones
- Viscous friction and torsional spring effects
- Encoder cable oscillation (Fourier-based, 5 harmonics)
- Hard mechanical angle limits on the arm (±2.2 rad)
- Domain randomization over all physical parameters for sim-to-real transfer

## RL Architecture

The SAC agent uses:

- **Actor**: Gaussian policy network outputting mean and log-std for continuous voltage commands
- **Critic**: Twin Q-networks to mitigate overestimation bias
- **Entropy**: Automatic temperature tuning targeting a fixed entropy budget
- **Training**: Replay buffer with batch sampling, soft target network updates (τ), and Adam optimizer

Hyperparameter evolution across parallel workers samples from Gaussian neighborhoods around the best-performing configuration.

## Control Strategies

| Controller | Description |
|---|---|
| **SAC (RL)** | Learned policy mapping states to voltage, trained in simulation with domain randomization |
| **PID** | Cascade controller with inner (motor angle) and outer (pendulum angle) loops |
| **LQR** | Linear state-feedback with swing-up energy controller and mode switching |

The `PIDvsRL/` directory contains experimental data comparing these controllers at voltage limits from 4V to 120V on real hardware.

## Dependencies

See `requirements.txt`. Core dependencies: PyTorch, NumPy, Matplotlib, Gymnasium, Numba, TensorBoard.

For hardware: `pyserial`, `PyQt5`, `pyqtgraph`.

## License

This project does not currently specify a license.
