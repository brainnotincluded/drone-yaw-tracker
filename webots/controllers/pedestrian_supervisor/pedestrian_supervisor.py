#!/usr/bin/env python3
"""
Webots Supervisor: broadcasts drone and pedestrian positions via TCP.

Port 9999 — position stream (50Hz JSON lines):
  {"drone": {"x","y","z","yaw"}, "human": {"x","y","z"}, "vec": {"dx","dy","dist","angle"}}

"vec" is the direction vector from drone to human:
  dx, dy  — components in world frame
  dist    — Euclidean distance (meters)
  angle   — bearing from drone to human (radians, world frame)
"""

import socket
import json
import math
import threading
import time


def run_server(drone_node, pedestrian_node, port=9999):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(5)
    server.settimeout(1.0)
    print(f"[Supervisor] Position server on port {port}")

    while True:
        try:
            conn, _ = server.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, drone_node, pedestrian_node),
                daemon=True,
            ).start()
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[Supervisor] Server error: {e}")
            break

    server.close()


def get_yaw_from_rotation(node):
    """Extract yaw from Webots rotation field [ax, ay, az, angle]."""
    rot = node.getField("rotation").getSFRotation()
    ax, ay, az, angle = rot
    # For rotation around Z axis: yaw = az * angle
    # General case: convert axis-angle to yaw
    if abs(az) > 0.9:
        return az * angle
    # Fallback: compute from orientation matrix
    import numpy as np
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    # Rotation matrix row 0, col 0 and row 1, col 0
    m00 = t * ax * ax + c
    m10 = t * ax * ay + az * s
    m01 = t * ax * ay - az * s
    m11 = t * ay * ay + c
    return math.atan2(m10, m00)


def handle_client(conn, drone_node, pedestrian_node):
    try:
        while True:
            dp = drone_node.getPosition()
            hp = pedestrian_node.getPosition()

            dx = hp[0] - dp[0]
            dy = hp[1] - dp[1]
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)

            yaw = get_yaw_from_rotation(drone_node)

            data = json.dumps({
                "drone": {
                    "x": round(dp[0], 3), "y": round(dp[1], 3),
                    "z": round(dp[2], 3), "yaw": round(yaw, 4)
                },
                "human": {
                    "x": round(hp[0], 3), "y": round(hp[1], 3),
                    "z": round(hp[2], 3)
                },
                "vec": {
                    "dx": round(dx, 3), "dy": round(dy, 3),
                    "dist": round(dist, 3),
                    "angle": round(angle, 4)
                }
            }) + "\n"

            conn.sendall(data.encode())
            time.sleep(0.02)  # 50Hz
    except Exception:
        pass
    finally:
        conn.close()


# ── Webots entry point ──────────────────────────────────────────────────
try:
    from controller import Supervisor

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    pedestrian = robot.getFromDef("PEDESTRIAN")
    drone = robot.getFromDef("DRONE")

    ok = True
    if pedestrian is None:
        print("[Supervisor] ERROR: DEF PEDESTRIAN not found!")
        ok = False
    if drone is None:
        print("[Supervisor] ERROR: DEF DRONE not found!")
        ok = False

    if ok:
        dp = drone.getPosition()
        hp = pedestrian.getPosition()
        print(f"[Supervisor] Drone:  {dp}")
        print(f"[Supervisor] Human:  {hp}")

        threading.Thread(
            target=run_server, args=(drone, pedestrian), daemon=True
        ).start()

        while robot.step(timestep) != -1:
            pass

except ImportError:
    print("[Supervisor] Not running inside Webots")
    import sys
    sys.exit(0)
