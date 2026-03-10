#!/usr/bin/env python3
"""Diagnostic: check if camera direction matches position data."""
import sys, os, time, math, struct, socket
import cv2
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, os.path.dirname(__file__))
from position_client import PositionClient

def get_frame(sock):
    try:
        hdr = b""
        while len(hdr) < 4:
            chunk = sock.recv(4 - len(hdr))
            if not chunk: return None
            hdr += chunk
        w, h = struct.unpack("<HH", hdr)
        data = b""
        need = w * h
        while len(data) < need:
            chunk = sock.recv(min(need - len(data), 65536))
            if not chunk: return None
            data += chunk
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    except:
        return None

def main():
    # Connect position client
    pos_client = PositionClient()
    if not pos_client.connect():
        return 1
    time.sleep(0.3)

    # Connect camera
    cam = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cam.settimeout(5.0)
    cam.connect(("127.0.0.1", 5599))

    # Connect drone
    m = mavutil.mavlink_connection("udp:127.0.0.1:14550")
    m.wait_heartbeat(timeout=15)
    m.mav.request_data_stream_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 50, 1)

    # GUIDED + Arm + Takeoff
    m.mav.set_mode_send(m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
    time.sleep(0.5)
    m.mav.command_long_send(m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    time.sleep(3)
    m.mav.command_long_send(m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 3.0)
    time.sleep(15)
    print("[+] At altitude")

    # Now slowly rotate 360 degrees, saving frames with position data
    TYPE_MASK = 0b010111000111
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("/home/webmaster/diag_rotate.mp4", fourcc, 10, (640, 640), True)

    print("[*] Rotating 360 degrees slowly, recording...")
    t0 = time.monotonic()
    duration = 30  # 30 seconds for full rotation = 12 deg/s = 0.21 rad/s
    yaw_rate = 0.21

    while time.monotonic() - t0 < duration:
        # Send yaw rate
        m.mav.set_position_target_local_ned_send(
            0, m.target_system, m.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED, TYPE_MASK,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float(yaw_rate))

        pos = pos_client.get()
        frame = get_frame(cam)

        if frame is not None and pos is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            h, w = bgr.shape[:2]

            # Crosshair
            cv2.line(bgr, (w//2-20, h//2), (w//2+20, h//2), (0,255,0), 1)
            cv2.line(bgr, (w//2, h//2-20), (w//2, h//2+20), (0,255,0), 1)

            rel_deg = math.degrees(pos.relative_angle)
            bearing_deg = math.degrees(pos.angle)
            yaw_deg = math.degrees(pos.drone_yaw)

            # Show all the data on frame
            texts = [
                f"rel_angle: {rel_deg:+.1f} deg",
                f"bearing:   {bearing_deg:+.1f} deg",
                f"drone_yaw: {yaw_deg:+.1f} deg",
                f"dist:      {pos.dist:.1f} m",
                f"dx={pos.dx:.1f} dy={pos.dy:.1f}",
                f"drone=({pos.drone_x:.1f},{pos.drone_y:.1f})",
                f"human=({pos.human_x:.1f},{pos.human_y:.1f})",
            ]
            for i, t in enumerate(texts):
                cv2.putText(bgr, t, (10, 25 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

            # Draw indicator based on relative angle (current mapping)
            px = int(w//2 + rel_deg / 60.0 * (w//2))
            px = max(10, min(w-10, px))
            cv2.arrowedLine(bgr, (px, h-60), (px, h-30), (0,0,255), 2)
            cv2.putText(bgr, "HUD", (px-15, h-65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            writer.write(bgr)

        time.sleep(0.1)

    # Stop
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED, TYPE_MASK,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    writer.release()
    cam.close()

    # Land
    m.mav.set_mode_send(m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
    time.sleep(10)
    pos_client.close()
    m.close()

    print("[+] Done! Video: /home/webmaster/diag_rotate.mp4")
    return 0

if __name__ == "__main__":
    sys.exit(main())
