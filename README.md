# Drone Yaw Tracker

RL-based drone yaw tracking: a simulated quadcopter learns to keep a walking human centered in its field of view by controlling yaw rate.

![Architecture](https://img.shields.io/badge/RL-PPO-blue) ![Simulator](https://img.shields.io/badge/Sim-Webots%20R2023b-green) ![Flight Controller](https://img.shields.io/badge/FC-ArduPilot%20SITL-orange)

## Overview

A PPO agent observes the relative angle to a human target and outputs yaw rate commands to track them. The system runs in Webots simulation with ArduPilot SITL.

**Architecture:**
```
Webots Supervisor → TCP:9999 → Position Client → RL Agent → MAVLink → ArduPilot → Webots Drone
                                                    ↑
                                              [rel_angle, angular_vel, prev_action]
```

**Results (88 episodes of training):**
| Metric | PID Baseline | RL (PPO) | RL + Roll Disturbance |
|--------|-------------|----------|----------------------|
| Avg error | 3.6° | 0.7° | 0.7° |
| < 3° | 53% | 98% | 98% |
| < 10° | 96% | 100% | 100% |

## Project Structure

```
├── src/                          # Core modules
│   ├── camera.py                 # Camera TCP stream receiver
│   ├── drone.py                  # MAVLink flight control helpers
│   ├── position_client.py        # Supervisor position stream client
│   └── hud.py                    # Video HUD overlay
├── scripts/                      # Runnable scripts
│   ├── train.py                  # PPO training
│   ├── eval.py                   # Evaluation with video recording
│   └── pid_baseline.py           # PID tracking baseline
├── webots/                       # Webots simulation files
│   ├── worlds/
│   │   └── iris_camera_human.wbt # World with Iris drone + pedestrian
│   └── controllers/
│       ├── pedestrian/           # Randomized pedestrian movement
│       └── pedestrian_supervisor/# Position broadcasting supervisor
├── models/                       # Trained model weights
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites
- [Webots R2023b](https://cyberbotics.com/)
- [ArduPilot SITL](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html) (ArduCopter)
- [MAVProxy](https://ardupilot.org/mavproxy/)
- Python 3.10+

### Install
```bash
pip install -r requirements.txt
```

### Start Simulation
```bash
# 1. Start Webots
webots --batch --mode=realtime webots/worlds/iris_camera_human.wbt

# 2. Start ArduCopter SITL
cd ~/ardupilot
build/sitl/bin/arducopter -S --model webots-python --speedup 1 \
  --defaults Tools/autotest/default_params/copter.parm,libraries/SITL/examples/Webots_Python/params/iris.parm \
  --sim-address=127.0.0.1 -I0

# 3. Start MAVProxy
mavproxy.py --master tcp:127.0.0.1:5760 --sitl 127.0.0.1:5501 \
  --out udp:127.0.0.1:14550 --daemon
```

## Usage

### Train
```bash
python scripts/train.py --timesteps 50000

# Resume training
python scripts/train.py --timesteps 100000 --resume models/ppo_tracker.zip
```

### Evaluate
```bash
# Basic evaluation (records video)
python scripts/eval.py --model models/ppo_tracker.zip --duration 60

# With lateral disturbance (tests robustness)
python scripts/eval.py --model models/ppo_tracker.zip --dist-vy 1.5 --duration 50
```

### PID Baseline
```bash
python scripts/pid_baseline.py --kp 1.5 --ki 0.2 --kd 0.8 --duration 60
```

## How It Works

### Observation Space (3D)
| Index | Value | Range |
|-------|-------|-------|
| 0 | Relative angle / π | [-1, 1] |
| 1 | Angular velocity / 2 | [-1, 1] |
| 2 | Previous action | [-1, 1] |

### Action Space (1D)
Yaw rate normalized to [-1, 1], scaled by `max_yaw_rate` (default 1.5 rad/s).

### Reward
```
r = -|relative_angle|
  + 1.0  if < 3°
  + 0.3  if < 10°
  + 0.1  if < 20°
  - 0.05 * |action - prev_action|  (smoothness)
```

### Training
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Network:** MLP [64, 64]
- **Episode:** 300 steps at 10Hz = 30s realtime
- **Control frequency:** 10Hz (observation + action each 100ms)

## License

MIT
