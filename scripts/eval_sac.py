#!/usr/bin/env python3
"""
Evaluate SAC yaw tracker: run inference, record video with HUD, log all telemetry.

Logs CSV with: timestamp, step, drone_x, drone_y, drone_z, drone_yaw,
human_x, human_y, human_z, distance, bearing, relative_angle, angular_velocity,
action_raw, action_clipped, yaw_rate_cmd, vy_cmd, reward, cumulative_reward

Usage:
    python eval_sac.py                                     # no-roll eval
    python eval_sac.py --vy 2.0                            # constant roll disturbance
    python eval_sac.py --vy-sweep 0,1,2,3 --duration 30    # sweep vy values
    python eval_sac.py --model ~/rl_models_sac4/sac_tracker.zip --duration 60
"""

import sys
import os
import time
import math
import csv
import argparse
from datetime import datetime

import cv2
import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(__file__))
from src.position_client import PositionClient
from src.camera import CameraReceiver
from src.hud import draw_crosshair, draw_target_indicator, draw_info, draw_disturbance
from src import drone


def run_eval(model, mav, pos_client, cam, args, vy=0.0, tag=""):
    """Run one evaluation segment. Returns (errors, csv_rows, output_path)."""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vy_tag = f"_vy{vy:+.1f}" if vy != 0 else "_noroll"
    base = f"eval_{ts}{vy_tag}{tag}"
    video_path = os.path.join(args.output_dir, f"{base}.mp4")
    csv_path = os.path.join(args.output_dir, f"{base}.csv")

    # Get frame dimensions
    test_frame = cam.get_frame()
    if test_frame is None:
        print("[-] No camera frame!")
        return [], [], None
    fh, fw = test_frame.shape

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, args.fps, (fw, fh), True)

    # CSV writer
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time_s", "step", "drone_x", "drone_y", "drone_z", "drone_yaw_deg",
        "human_x", "human_y", "human_z", "distance_m", "bearing_deg",
        "relative_angle_deg", "angular_velocity_dps", "action_raw", "action_clipped",
        "yaw_rate_cmd_dps", "vy_cmd", "reward", "cumulative_reward",
        "within_3deg", "within_10deg"
    ])

    control_dt = 1.0 / args.control_freq
    prev_action = 0.0
    prev_rel_angle = 0.0
    errors = []
    cumulative_reward = 0.0

    label = f"SAC vy={vy:+.1f}" if vy != 0 else "SAC no-roll"
    print(f"\n  [{label}] Recording {args.duration}s → {video_path}")

    t_start = time.monotonic()
    step = 0

    try:
        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            elapsed = now - t_start

            pos = pos_client.get()

            action_raw = 0.0
            action_clipped = 0.0
            yaw_rate_cmd = 0.0
            ang_vel = 0.0
            reward = 0.0

            if pos and pos.dist > 0.1:
                rel = pos.relative_angle
                diff = rel - prev_rel_angle
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                ang_vel = diff / max(control_dt, 0.001)

                obs = np.array([
                    np.clip(rel / math.pi, -1, 1),
                    np.clip(ang_vel / 2.0, -1, 1),
                    prev_action,
                    np.clip(vy / 3.0, -1, 1),  # vy_max=3.0
                ], dtype=np.float32)

                prev_rel_angle = rel

                # Model inference
                action, _ = model.predict(obs, deterministic=True)
                action_raw = float(action[0])
                action_clipped = float(np.clip(action_raw, -1, 1))

                # Rate limit (same as training)
                max_delta = 0.3
                action_clipped = float(np.clip(
                    action_clipped,
                    prev_action - max_delta,
                    prev_action + max_delta
                ))

                yaw_rate_cmd = -action_clipped * args.max_yaw_rate

                # Send command with vy
                drone.send_yaw_rate(mav, yaw_rate_cmd, vy=vy, drone_yaw=pos.drone_yaw)
                prev_action = action_clipped

                # Compute reward (same as training for comparison)
                abs_err = abs(rel)
                deg = math.degrees(abs_err)
                reward = -abs_err
                reward += 2.0 * math.exp(-(deg / 5.0) ** 2)
                reward -= 0.2 * abs(action_clipped - prev_action)
                reward -= 0.15 * action_clipped ** 2
                if deg < 30:
                    closeness = 1.0 - deg / 30.0
                    reward -= 0.4 * (ang_vel ** 2) * closeness
                cumulative_reward += reward

                err_deg = abs(math.degrees(rel))
                errors.append(err_deg)

                # CSV row
                csv_writer.writerow([
                    f"{elapsed:.3f}", step,
                    f"{pos.drone_x:.3f}", f"{pos.drone_y:.3f}", f"{pos.drone_z:.3f}",
                    f"{math.degrees(pos.drone_yaw):.2f}",
                    f"{pos.human_x:.3f}", f"{pos.human_y:.3f}", f"{pos.human_z:.3f}",
                    f"{pos.dist:.2f}", f"{math.degrees(pos.angle):.2f}",
                    f"{math.degrees(rel):.2f}",
                    f"{math.degrees(ang_vel):.1f}",
                    f"{action_raw:.4f}", f"{action_clipped:.4f}",
                    f"{math.degrees(yaw_rate_cmd):.1f}",
                    f"{vy:.2f}", f"{reward:.3f}", f"{cumulative_reward:.2f}",
                    int(err_deg < 3), int(err_deg < 10)
                ])

            # Camera frame + HUD
            frame = cam.get_frame()
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                draw_crosshair(bgr)
                if pos:
                    rel_deg = math.degrees(pos.relative_angle)
                    draw_target_indicator(bgr, pos.relative_angle)
                    draw_info(bgr, [
                        f"drone=({pos.drone_x:.1f},{pos.drone_y:.1f},{pos.drone_z:.1f})",
                        f"human=({pos.human_x:.1f},{pos.human_y:.1f})",
                        f"dist={pos.dist:.1f}m",
                        f"rel={rel_deg:+.1f} deg  ang_vel={math.degrees(ang_vel):+.0f} dps",
                        f"action={action_clipped:+.3f}  yaw={math.degrees(yaw_rate_cmd):+.1f} dps",
                        f"T={elapsed:.1f}s  [{label}]",
                    ])
                    if vy != 0:
                        draw_disturbance(bgr, vy)
                writer.write(bgr)

            step += 1

            # Status every 5s
            if step % (args.control_freq * 5) == 0 and errors:
                recent = errors[-100:]
                avg = sum(recent) / len(recent)
                p3 = 100 * sum(1 for e in errors if e < 3) / len(errors)
                p10 = 100 * sum(1 for e in errors if e < 10) / len(errors)
                print(f"    T={elapsed:5.1f}s  err={errors[-1]:5.1f}°  "
                      f"avg={avg:4.1f}  <3°={p3:4.1f}%  <10°={p10:4.1f}%  "
                      f"R={cumulative_reward:+.0f}")

            sleep_time = control_dt - (time.monotonic() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n  [!] Interrupted")

    # Stop
    drone.send_yaw_rate(mav, 0, vy=0)
    writer.release()
    csv_file.close()

    # Summary
    total = len(errors)
    if total > 0:
        avg_err = sum(errors) / total
        p3 = 100 * sum(1 for e in errors if e < 3) / total
        p10 = 100 * sum(1 for e in errors if e < 10) / total
        med = sorted(errors)[total // 2]
        mx = max(errors)
        p95 = sorted(errors)[int(total * 0.95)]
        print(f"\n  --- {label} RESULTS ---")
        print(f"  Steps:     {total}")
        print(f"  Avg err:   {avg_err:.2f}°")
        print(f"  Median:    {med:.2f}°")
        print(f"  P95:       {p95:.2f}°")
        print(f"  Max:       {mx:.2f}°")
        print(f"  < 3°:      {p3:.1f}%")
        print(f"  < 10°:     {p10:.1f}%")
        print(f"  Cum. R:    {cumulative_reward:+.1f}")
        print(f"  Video:     {video_path}")
        print(f"  CSV:       {csv_path}")

    return errors, csv_path, video_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC yaw tracker")
    parser.add_argument("--model", type=str,
                        default="/home/webmaster/rl_models_sac4/sac_tracker.zip")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--max-yaw-rate", type=float, default=1.5)
    parser.add_argument("--altitude", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--vy", type=float, default=0.0,
                        help="Constant lateral velocity for roll disturbance")
    parser.add_argument("--vy-sweep", type=str, default=None,
                        help="Comma-separated vy values to sweep, e.g. '0,1,2,3'")
    parser.add_argument("--output-dir", type=str, default="/home/webmaster/eval_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"[*] Loading model: {args.model}")
    model = SAC.load(args.model)

    # Connect position client
    print("[*] Connecting position client...")
    pos_client = PositionClient()
    if not pos_client.connect():
        return 1
    time.sleep(0.3)

    # Connect camera
    print("[*] Connecting camera...")
    cam = CameraReceiver()
    cam.connect()

    # Connect drone
    print("[*] Connecting drone...")
    mav = drone.connect()
    print(f"[+] Drone system {mav.target_system}")

    # Takeoff
    print("[*] GUIDED + Arm + Takeoff...")
    drone.set_guided(mav)
    if not drone.arm(mav):
        print("[-] Arm failed")
        return 1
    drone.takeoff(mav, args.altitude)
    print(f"[*] Climbing to {args.altitude}m...")
    time.sleep(15)
    print("[+] At altitude")

    print(f"\n{'='*60}")
    print(f"  SAC EVAL — model: {os.path.basename(args.model)}")
    print(f"  Control: {args.control_freq}Hz | max yaw: {args.max_yaw_rate} rad/s")
    print(f"{'='*60}")

    if args.vy_sweep:
        # Run multiple segments with different vy
        vy_values = [float(v) for v in args.vy_sweep.split(",")]
        for vy in vy_values:
            # Reset position between segments
            drone.reset_position()
            time.sleep(0.5)
            run_eval(model, mav, pos_client, cam, args, vy=vy)
            time.sleep(1.0)
    else:
        run_eval(model, mav, pos_client, cam, args, vy=args.vy)

    # Land
    print("\n[*] Landing...")
    drone.send_yaw_rate(mav, 0, vy=0)
    drone.land(mav)
    time.sleep(10)
    pos_client.close()
    cam.close()
    mav.close()
    print("[+] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
