#!/usr/bin/env python3
"""
Evaluate trained RL model: run inference + record video with HUD.
"""

import sys
import os
import time
import math
import struct
import socket
import argparse

import cv2
import numpy as np
from pymavlink import mavutil
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))
from position_client import PositionClient


# ── Camera receiver ──────────────────────────────────────────────────────

class CameraReceiver:
    def __init__(self, host="127.0.0.1", port=5599):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.host, self.port))

    def get_frame(self):
        try:
            hdr = b""
            while len(hdr) < 4:
                chunk = self.sock.recv(4 - len(hdr))
                if not chunk:
                    return None
                hdr += chunk
            w, h = struct.unpack("<HH", hdr)
            data = b""
            need = w * h
            while len(data) < need:
                chunk = self.sock.recv(min(need - len(data), 65536))
                if not chunk:
                    return None
                data += chunk
            return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
        except Exception:
            return None

    def close(self):
        if self.sock:
            self.sock.close()


# ── MAVLink helpers ──────────────────────────────────────────────────────

TYPE_MASK_YAW_RATE = 0b010111000111


def connect_drone(conn_str="udp:127.0.0.1:14550"):
    m = mavutil.mavlink_connection(conn_str)
    m.wait_heartbeat(timeout=15)
    m.mav.request_data_stream_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 50, 1)
    return m


def set_guided(m):
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
    time.sleep(0.5)


def arm(m):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    for _ in range(30):
        msg = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            return True
    return False


def takeoff(m, alt=5.0):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt)


def send_yaw_rate(m, yaw_rate):
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        TYPE_MASK_YAW_RATE,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, float(yaw_rate))


def land(m):
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)


# ── HUD overlay ──────────────────────────────────────────────────────────

def draw_hud(frame_bgr, pos, action, yaw_rate_cmd, step_num, elapsed):
    h, w = frame_bgr.shape[:2]

    # Center crosshair
    cv2.line(frame_bgr, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 1)
    cv2.line(frame_bgr, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 1)

    if pos:
        rel_deg = math.degrees(pos.relative_angle)
        px_offset = int(rel_deg / 60.0 * (w / 2))
        indicator_x = max(10, min(w - 10, w // 2 + px_offset))
        cv2.arrowedLine(frame_bgr, (indicator_x, 30), (indicator_x, 60), (0, 0, 255), 2)
        cv2.circle(frame_bgr, (indicator_x, 30), 6, (0, 0, 255), -1)

        info = [
            f"dist={pos.dist:.1f}m",
            f"rel={rel_deg:+.1f} deg",
            f"action={action:+.2f}",
            f"yaw_cmd={math.degrees(yaw_rate_cmd):+.1f} deg/s",
            f"T={elapsed:.1f}s  [RL]",
        ]
        for i, txt in enumerate(info):
            cv2.putText(frame_bgr, txt, (10, h - 15 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return frame_bgr


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/home/webmaster/rl_models/ppo_tracker.zip")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--max-yaw-rate", type=float, default=1.5)
    parser.add_argument("--altitude", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"/home/webmaster/rl_eval_{ts}.mp4"

    # Load model
    print(f"[*] Loading model: {args.model}")
    model = PPO.load(args.model)

    # Connect
    print("[*] Connecting position client...")
    pos_client = PositionClient()
    if not pos_client.connect():
        return 1
    time.sleep(0.3)

    print("[*] Connecting camera...")
    cam = CameraReceiver()
    cam.connect()
    test_frame = cam.get_frame()
    if test_frame is None:
        print("[-] No camera frame!")
        return 1
    fh, fw = test_frame.shape
    print(f"[+] Camera: {fw}x{fh}")

    print("[*] Connecting drone...")
    mav = connect_drone()
    print(f"[+] Drone system {mav.target_system}")

    # Takeoff
    print("[*] GUIDED + Arm + Takeoff...")
    set_guided(mav)
    if not arm(mav):
        print("[-] Arm failed")
        return 1
    takeoff(mav, args.altitude)

    print(f"[*] Climbing to {args.altitude}m...")
    time.sleep(15)
    print("[+] At altitude")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (fw, fh), True)
    print(f"[+] Recording to {args.output}")

    # Eval loop
    control_dt = 1.0 / args.control_freq
    prev_action = 0.0
    prev_rel_angle = 0.0
    errors = []
    in_3 = 0
    in_10 = 0

    print(f"\n{'='*60}")
    print(f"  RL EVAL — {args.duration}s")
    print(f"{'='*60}\n")

    t_start = time.monotonic()
    step = 0

    try:
        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            elapsed = now - t_start

            pos = pos_client.get()

            # Build observation
            action_val = 0.0
            yaw_rate_cmd = 0.0
            if pos and pos.dist > 0.1:
                rel = pos.relative_angle
                diff = rel - prev_rel_angle
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                ang_vel = diff / max(control_dt, 0.001)

                obs = np.array([
                    np.clip(rel / math.pi, -1, 1),
                    np.clip(ang_vel / 2.0, -1, 1),
                    prev_action
                ], dtype=np.float32)

                prev_rel_angle = rel

                # Model inference
                action, _ = model.predict(obs, deterministic=True)
                action_val = float(np.clip(action[0], -1, 1))
                yaw_rate_cmd = -action_val * args.max_yaw_rate
                send_yaw_rate(mav, yaw_rate_cmd)
                prev_action = action_val

                abs_deg = abs(math.degrees(rel))
                errors.append(abs_deg)
                if abs_deg < 3:
                    in_3 += 1
                if abs_deg < 10:
                    in_10 += 1

            # Camera frame
            frame = cam.get_frame()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                draw_hud(frame_bgr, pos, action_val, yaw_rate_cmd, step, elapsed)
                writer.write(frame_bgr)

            step += 1

            # Status every 5s
            if step % (args.control_freq * 5) == 0 and errors:
                avg = sum(errors[-100:]) / min(len(errors), 100)
                w3 = 100 * in_3 / len(errors)
                w10 = 100 * in_10 / len(errors)
                rel_deg = math.degrees(pos.relative_angle) if pos else 0
                print(f"  T={elapsed:5.1f}s  err={rel_deg:+6.1f}°  "
                      f"avg={avg:5.1f}  <3°={w3:4.1f}%  <10°={w10:4.1f}%")

            sleep_time = control_dt - (time.monotonic() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[!] Interrupted")

    send_yaw_rate(mav, 0)
    writer.release()
    cam.close()

    print(f"\n[*] Landing...")
    land(mav)
    time.sleep(10)

    pos_client.close()
    mav.close()

    total = len(errors)
    if total > 0:
        avg_err = sum(errors) / total
        w3 = 100 * in_3 / total
        w10 = 100 * in_10 / total
        print(f"\n{'='*60}")
        print(f"  RL EVAL RESULTS")
        print(f"{'='*60}")
        print(f"  Duration:      {args.duration:.0f}s ({total} steps)")
        print(f"  Avg error:     {avg_err:.1f}°")
        print(f"  < 3°:          {w3:.1f}%")
        print(f"  < 10°:         {w10:.1f}%")
        print(f"  Video:         {args.output}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
