#!/usr/bin/env python3
"""Evaluate trained RL model with optional lateral disturbance.

Records video with HUD overlay showing tracking performance.

Usage:
    python scripts/eval.py --model models/ppo_tracker.zip
    python scripts/eval.py --model models/ppo_tracker.zip --dist-vy 1.5
"""

import sys
import os
import time
import math
import argparse

import cv2
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.position_client import PositionClient
from src.camera import CameraReceiver
from src.hud import draw_tracking_hud
from src import drone


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL yaw tracker")
    parser.add_argument("--model", default="models/ppo_tracker.zip")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--altitude", type=float, default=2.5)
    parser.add_argument("--max-yaw-rate", type=float, default=1.5)
    parser.add_argument("--dist-vy", type=float, default=0.0,
                        help="Constant lateral velocity disturbance (m/s)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"eval_{ts}.mp4"

    print(f"[*] Loading model: {args.model}")
    model = PPO.load(args.model)

    pos_client = PositionClient()
    if not pos_client.connect():
        return 1
    time.sleep(0.3)

    cam = CameraReceiver()
    cam.connect()
    test = cam.get_frame()
    if test is None:
        print("[-] No camera!")
        return 1
    fh, fw = test.shape
    print(f"[+] Camera: {fw}x{fh}")

    mav = drone.connect()
    print(f"[+] Drone system {mav.target_system}")

    print("[*] GUIDED + Arm + Takeoff...")
    drone.set_guided(mav)
    if not drone.arm(mav):
        print("[-] Arm failed")
        return 1
    drone.takeoff(mav, args.altitude)
    print(f"[*] Climbing to {args.altitude}m...")
    time.sleep(12)
    print("[+] At altitude")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, 30, (fw, fh), True)
    print(f"[+] Recording to {args.output}")

    control_dt = 0.1  # 10Hz (matches training)
    prev_action = 0.0
    prev_rel = 0.0
    errors = []
    in_3 = in_10 = 0
    last_action_val = 0.0
    last_yaw_rate_cmd = 0.0
    last_pos = None

    has_dist = args.dist_vy != 0.0
    label = "RL+DIST" if has_dist else "RL"

    print(f"\n{'=' * 60}")
    print(f"  RL EVAL — {args.duration}s")
    if has_dist:
        print(f"  Lateral disturbance: vy={args.dist_vy} m/s")
    print(f"{'=' * 60}\n")

    t_start = time.monotonic()
    last_ctrl = t_start
    step = 0

    try:
        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            elapsed = now - t_start

            dist_active = has_dist and elapsed > 3.0
            vy = args.dist_vy if dist_active else 0.0

            if now - last_ctrl >= control_dt:
                last_ctrl = now
                pos = pos_client.get()
                last_pos = pos

                if pos and pos.dist > 0.1:
                    rel = pos.relative_angle
                    diff = (rel - prev_rel + math.pi) % (2 * math.pi) - math.pi
                    ang_vel = diff / max(control_dt, 0.001)

                    obs = np.array([
                        np.clip(rel / math.pi, -1, 1),
                        np.clip(ang_vel / 2.0, -1, 1),
                        prev_action
                    ], dtype=np.float32)
                    prev_rel = rel

                    action, _ = model.predict(obs, deterministic=True)
                    last_action_val = float(np.clip(action[0], -1, 1))
                    last_yaw_rate_cmd = -last_action_val * args.max_yaw_rate
                    drone.send_yaw_rate(mav, last_yaw_rate_cmd, vy=vy)
                    prev_action = last_action_val

                    abs_deg = abs(math.degrees(rel))
                    errors.append(abs_deg)
                    if abs_deg < 3: in_3 += 1
                    if abs_deg < 10: in_10 += 1

                step += 1
                if step % 50 == 0 and errors:
                    avg = sum(errors[-100:]) / min(len(errors), 100)
                    rel_deg = math.degrees(pos.relative_angle) if pos else 0
                    print(f"  T={elapsed:5.1f}s  err={rel_deg:+6.1f}°  avg={avg:5.1f}°")

            frame = cam.get_frame()
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                draw_tracking_hud(bgr, last_pos, last_action_val,
                                  last_yaw_rate_cmd, elapsed, label=label,
                                  dist_vy=vy, dist_active=dist_active)
                writer.write(bgr)

    except KeyboardInterrupt:
        print("\n[!] Interrupted")

    drone.send_yaw_rate(mav, 0)
    writer.release()
    cam.close()

    print("[*] Landing...")
    drone.land(mav)
    time.sleep(10)
    pos_client.close()
    mav.close()

    total = len(errors)
    if total > 0:
        print(f"\n{'=' * 60}")
        print(f"  EVAL RESULTS")
        print(f"{'=' * 60}")
        print(f"  Duration:    {args.duration:.0f}s ({total} steps)")
        print(f"  Avg error:   {sum(errors) / total:.1f}°")
        print(f"  < 3°:        {100 * in_3 / total:.1f}%")
        print(f"  < 10°:       {100 * in_10 / total:.1f}%")
        if has_dist:
            print(f"  Disturbance: vy={args.dist_vy} m/s")
        print(f"  Video:       {args.output}")
        print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
