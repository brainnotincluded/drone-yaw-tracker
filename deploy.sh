#!/bin/bash
# Deploy drone yaw tracker to a server with ArduPilot SITL already set up
# Usage: ./deploy.sh [user@host]
set -e

HOST=${1:-webmaster@95.143.188.159}
WEBOTS_DIR="ardupilot/libraries/SITL/examples/Webots_Python"

echo "=== Deploying to $HOST ==="

# Core scripts
scp scripts/train_sac_v5.py $HOST:~/train_sac_v5.py
scp scripts/eval_sac.py $HOST:~/eval_sac.py
scp scripts/position_client.py $HOST:~/position_client.py

# Source modules
ssh $HOST 'mkdir -p ~/src ~/rl_models_v5'
scp src/drone.py $HOST:~/src/drone.py
scp src/yaw_controller.py $HOST:~/yaw_controller.py

# Webots files
ssh $HOST "mkdir -p ~/$WEBOTS_DIR/worlds ~/$WEBOTS_DIR/controllers/pedestrian_supervisor"
scp webots/worlds/iris_camera_human.wbt $HOST:~/$WEBOTS_DIR/worlds/
scp webots/controllers/pedestrian_supervisor/pedestrian_supervisor.py $HOST:~/$WEBOTS_DIR/controllers/pedestrian_supervisor/
scp webots/params/iris.parm $HOST:~/$WEBOTS_DIR/params/

# Models
scp models/sac_loiter_best.zip $HOST:~/rl_models_v5/sac_loiter_best.zip
scp models/sac_loiter.zip $HOST:~/rl_models_v5/sac_loiter.zip

echo "=== Deploy complete ==="
echo ""
echo "Next steps on server:"
echo "  1. Install Webots: sudo apt install webots (or download from cyberbotics.com)"
echo "  2. Install Xvfb: sudo apt install xvfb"
echo "  3. Install Python packages:"
echo "     pip3 install stable-baselines3 torch gymnasium pymavlink numpy"
echo "  4. Start Xvfb: Xvfb :99 -screen 0 1024x768x24 &"
echo "  5. Start sim stack (see sim.sh)"
