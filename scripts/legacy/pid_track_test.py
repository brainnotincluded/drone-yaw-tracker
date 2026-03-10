#!/usr/bin/env python3
"""
PID yaw tracking test: drone tracks human using position_client + camera video.

1. Takeoff
2. PID controls yaw to keep relative_angle ≈ 0
3. Records camera feed as MP4
4. Prints tracking stats
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

sys.path.insert(0, os.path.dirname(__file__))
from position_client import PositionClient


# ── PID ──────────────────────────────────────────────────────────────────

class PID:
    def __init__(self, kp, ki, kd, output_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = output_limit
        self.integral = 0.0
        self.prev_error = None

    def step(self, error, dt):
        self.integral += error * dt
        self.integral = max(-self.limit, min(self.limit, self.integral))

        deriv = 0.0
        if self.prev_error is not None and dt > 0:
            deriv = (error - self.prev_error) / dt
        self.prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        return max(-self.limit, min(self.limit, out))

    def reset(self):
        self.integral = 0.0
        self.prev_error = None


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


# ── Drone MAVLink ────────────────────────────────────────────────────────

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

def draw_hud(frame_bgr, pos, pid_out, yaw_rate_cmd, step_num, elapsed):
    h, w = frame_bgr.shape[:2]

    # Center crosshair
    cv2.line(frame_bgr, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 0), 1)
    cv2.line(frame_bgr, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 0), 1)

    # Direction indicator: where human is relative to drone heading
    if pos:
        rel_deg = math.degrees(pos.relative_angle)
        # Map relative angle to pixel offset on screen
        px_offset = int(rel_deg / 60.0 * (w / 2))  # assume ~60 deg half-FOV
        indicator_x = w // 2 + px_offset
        indicator_x = max(10, min(w - 10, indicator_x))
        cv2.arrowedLine(frame_bgr, (indicator_x, 30), (indicator_x, 60), (0, 0, 255), 2)
        cv2.circle(frame_bgr, (indicator_x, 30), 6, (0, 0, 255), -1)

        # Text info
        info = [
            f"dist={pos.dist:.1f}m",
            f"rel={rel_deg:+.1f} deg",
            f"pid={pid_out:+.2f}",
            f"yaw_cmd={math.degrees(yaw_rate_cmd):+.1f} deg/s",
            f"T={elapsed:.1f}s",
        ]
        for i, txt in enumerate(info):
            cv2.putText(frame_bgr, txt, (10, h - 15 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return frame_bgr


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60, help="Tracking duration (s)")
    parser.add_argument("--altitude", type=float, default=5.0)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--ki", type=float, default=0.1)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0, help="Max yaw rate (rad/s)")
    parser.add_argument("--output", type=str, default=None, help="Video output path")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.output is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"/home/webmaster/pid_tracking_{ts}.mp4"

    # Connect everything
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
    for _ in range(80):
        msg = mav.recv_match(type='VFR_HUD', blocking=True, timeout=1)
        if msg and msg.alt >= 584 + args.altitude - 0.5:
            break
        time.sleep(0.3)
    print("[+] At altitude")
    time.sleep(1)

    # Setup PID and video
    pid = PID(args.kp, args.ki, args.kd, output_limit=1.0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (fw, fh), True)
    print(f"[+] Recording to {args.output}")

    # Tracking loop
    print(f"\n{'='*60}")
    print(f"  PID TRACKING  kp={args.kp} ki={args.ki} kd={args.kd}")
    print(f"  Duration: {args.duration}s  Max yaw rate: {args.max_yaw_rate} rad/s")
    print(f"{'='*60}\n")

    t_start = time.monotonic()
    last_t = t_start
    step = 0
    errors = []
    in_window_3 = 0
    in_window_10 = 0

    try:
        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            dt = now - last_t
            last_t = now
            elapsed = now - t_start

            # Get positions
            pos = pos_client.get()

            # PID on relative angle
            pid_out = 0.0
            yaw_rate_cmd = 0.0
            if pos and pos.dist > 0.5:
                error = pos.relative_angle  # radians, >0 = human to the right
                pid_out = pid.step(error, max(dt, 0.001))
                yaw_rate_cmd = -pid_out * args.max_yaw_rate
                send_yaw_rate(mav, yaw_rate_cmd)

                abs_deg = abs(math.degrees(error))
                errors.append(abs_deg)
                if abs_deg < 3:
                    in_window_3 += 1
                if abs_deg < 10:
                    in_window_10 += 1

            # Camera frame
            frame = cam.get_frame()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                draw_hud(frame_bgr, pos, pid_out, yaw_rate_cmd, step, elapsed)
                writer.write(frame_bgr)

            step += 1

            # Status every 5s
            if step % 150 == 0 and len(errors) > 0:
                avg_err = sum(errors[-100:]) / min(len(errors), 100)
                w3 = 100 * in_window_3 / len(errors)
                w10 = 100 * in_window_10 / len(errors)
                rel_deg = math.degrees(pos.relative_angle) if pos else 0
                print(f"  T={elapsed:5.1f}s  err={rel_deg:+6.1f} deg  "
                      f"avg={avg_err:5.1f}  <3deg={w3:4.1f}%  <10deg={w10:4.1f}%  "
                      f"dist={pos.dist:.1f}m" if pos else "")

            # ~30 Hz control rate
            sleep_time = 1.0 / args.fps - (time.monotonic() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[!] Interrupted")

    # Stop and land
    send_yaw_rate(mav, 0)
    writer.release()
    cam.close()

    print(f"\n[*] Landing...")
    land(mav)
    time.sleep(10)

    pos_client.close()
    mav.close()

    # Stats
    total = len(errors)
    if total > 0:
        avg_err = sum(errors) / total
        w3 = 100 * in_window_3 / total
        w10 = 100 * in_window_10 / total
        print(f"\n{'='*60}")
        print(f"  TRACKING RESULTS")
        print(f"{'='*60}")
        print(f"  Duration:      {args.duration:.0f}s ({total} steps)")
        print(f"  PID gains:     kp={args.kp}  ki={args.ki}  kd={args.kd}")
        print(f"  Avg error:     {avg_err:.1f} deg")
        print(f"  < 3 deg:       {w3:.1f}%")
        print(f"  < 10 deg:      {w10:.1f}%")
        print(f"  Video:         {args.output}")
        print(f"{'='*60}")
    else:
        print("[!] No tracking data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
