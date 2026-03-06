#!/usr/bin/env python3
"""PID yaw tracking baseline for comparison with RL.

Usage:
    python scripts/pid_baseline.py --duration 60 --kp 1.5 --ki 0.2 --kd 0.8
"""

import sys
import os
import time
import math
import argparse

import cv2
from pymavlink import mavutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.position_client import PositionClient
from src.camera import CameraReceiver
from src.hud import draw_tracking_hud
from src import drone


class PID:
    def __init__(self, kp, ki, kd, output_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = output_limit
        self.integral = 0.0
        self.prev_error = None

    def step(self, error, dt):
        self.integral = max(-self.limit, min(self.limit,
                            self.integral + error * dt))
        deriv = 0.0
        if self.prev_error is not None and dt > 0:
            deriv = (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        return max(-self.limit, min(self.limit, out))


def main():
    parser = argparse.ArgumentParser(description="PID yaw tracking baseline")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--altitude", type=float, default=5.0)
    parser.add_argument("--kp", type=float, default=1.5)
    parser.add_argument("--ki", type=float, default=0.2)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"pid_{ts}.mp4"

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

    mav = drone.connect()
    print(f"[+] Drone system {mav.target_system}")

    drone.set_guided(mav)
    if not drone.arm(mav):
        print("[-] Arm failed")
        return 1
    drone.takeoff(mav, args.altitude)
    print(f"[*] Climbing to {args.altitude}m...")
    time.sleep(15)
    print("[+] At altitude")

    pid = PID(args.kp, args.ki, args.kd)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, 30, (fw, fh), True)

    print(f"\n  PID TRACKING  kp={args.kp} ki={args.ki} kd={args.kd}\n")

    t_start = time.monotonic()
    last_t = t_start
    errors = []
    in_3 = in_10 = 0
    step = 0

    try:
        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            dt = now - last_t
            last_t = now
            elapsed = now - t_start

            pos = pos_client.get()
            pid_out = yaw_rate_cmd = 0.0

            if pos and pos.dist > 0.5:
                error = pos.relative_angle
                pid_out = pid.step(error, max(dt, 0.001))
                yaw_rate_cmd = -pid_out * args.max_yaw_rate
                drone.send_yaw_rate(mav, yaw_rate_cmd)

                abs_deg = abs(math.degrees(error))
                errors.append(abs_deg)
                if abs_deg < 3: in_3 += 1
                if abs_deg < 10: in_10 += 1

            frame = cam.get_frame()
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                draw_tracking_hud(bgr, pos, pid_out, yaw_rate_cmd,
                                  elapsed, label="PID")
                writer.write(bgr)

            step += 1
            if step % 150 == 0 and errors:
                avg = sum(errors[-100:]) / min(len(errors), 100)
                rel_deg = math.degrees(pos.relative_angle) if pos else 0
                print(f"  T={elapsed:5.1f}s  err={rel_deg:+6.1f}°  avg={avg:5.1f}°")

            time.sleep(max(0, 1.0 / 30 - (time.monotonic() - now)))

    except KeyboardInterrupt:
        print("\n[!] Interrupted")

    drone.send_yaw_rate(mav, 0)
    writer.release()
    cam.close()
    drone.land(mav)
    time.sleep(10)
    pos_client.close()
    mav.close()

    total = len(errors)
    if total > 0:
        print(f"\n  RESULTS: avg={sum(errors)/total:.1f}°  "
              f"<3°={100*in_3/total:.1f}%  <10°={100*in_10/total:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
