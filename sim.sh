#!/bin/bash
# Start the full simulation stack for drone yaw tracker training
# Must be run on the server with Webots, ArduPilot SITL, and MAVProxy installed
set -e

WEBOTS_DIR="$HOME/ardupilot/libraries/SITL/examples/Webots_Python"

echo "=== Cleaning up old processes ==="
killall -9 arducopter 2>/dev/null || true
killall -9 webots webots-bin 2>/dev/null || true
pkill -9 -f mavproxy 2>/dev/null || true
pkill -9 -f sim_vehicle 2>/dev/null || true
pkill -9 -f train_sac 2>/dev/null || true
sleep 2
fuser -k 5760/tcp 2>/dev/null || true

echo "=== Clearing Webots cache ==="
rm -rf /tmp/webots*

echo "=== Starting Webots ==="
export DISPLAY=:99
cd "$WEBOTS_DIR"
nohup webots --mode=fast --no-rendering worlds/iris_camera_human.wbt > /tmp/webots.log 2>&1 &
echo "Webots PID=$! — waiting 12s..."
sleep 12

echo "=== Starting ArduCopter SITL ==="
cd ~/ardupilot
nohup Tools/autotest/sim_vehicle.py -v ArduCopter -f webots-python \
    --add-param-file=libraries/SITL/examples/Webots_Python/params/iris.parm \
    --speedup 2 --no-rebuild --no-mavproxy > /tmp/ardu.log 2>&1 &
echo "ArduCopter PID=$! — waiting 10s..."
sleep 10

echo "=== Starting MAVProxy ==="
source ~/venv-ardupilot/bin/activate
nohup mavproxy.py --master tcp:127.0.0.1:5760 --sitl 127.0.0.1:5501 \
    --out udp:127.0.0.1:14550 --daemon > /tmp/mavproxy.log 2>&1 &
echo "MAVProxy PID=$! — waiting 6s..."
sleep 6

echo ""
echo "=== Status ==="
W=$(pgrep -c webots-bin 2>/dev/null || echo 0)
A=$(pgrep -c arducopter 2>/dev/null || echo 0)
M=$(pgrep -c mavproxy 2>/dev/null || echo 0)
echo "Webots: $W | ArduCopter: $A | MAVProxy: $M"

if [ "$W" -ge 1 ] && [ "$A" -ge 1 ] && [ "$M" -ge 1 ]; then
    echo "=== All systems GO ==="
    echo ""
    echo "To start training:"
    echo "  nohup python3 -u ~/train_sac_v5.py --timesteps 600000 > /tmp/sac_v5_train.log 2>&1 &"
    echo ""
    echo "To resume from checkpoint:"
    echo "  nohup python3 -u ~/train_sac_v5.py --timesteps 600000 --resume ~/rl_models_v5/sac_loiter_best.zip > /tmp/sac_v5_train.log 2>&1 &"
else
    echo "=== ERROR: Some processes failed to start ==="
    echo "Check /tmp/webots.log, /tmp/ardu.log, /tmp/mavproxy.log"
fi
