#!/usr/bin/env python3
"""Test drone: arm, takeoff, yaw rotation, land."""
import sys
import time
from pymavlink import mavutil

CONN = "udp:127.0.0.1:14550"

def main():
    print(f"[*] Connecting to {CONN}...")
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat(timeout=15)
    print(f"[+] Heartbeat from system {master.target_system}")

    # Request data streams
    master.mav.request_data_stream_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    time.sleep(1)

    # Set GUIDED mode
    print("[*] Setting GUIDED mode...")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
    time.sleep(1)

    # Arm
    print("[*] Arming...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)

    for _ in range(30):
        msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("[+] Armed!")
            break
    else:
        print("[-] Failed to arm")
        return 1

    # Takeoff to 5m
    print("[*] Takeoff to 5m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, 5.0)

    # Wait for altitude
    print("[*] Climbing...")
    for _ in range(60):
        msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1)
        if msg:
            print(f"    Alt: {msg.alt:.1f}m  Heading: {msg.heading}")
            if msg.alt >= 4.5:
                print("[+] Reached altitude!")
                break
        time.sleep(0.5)

    # Hover 3 seconds
    print("[*] Hovering 3s...")
    time.sleep(3)

    # Yaw rotation test: rotate 90 degrees right
    print("[*] Rotating 90 deg RIGHT (CW)...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        90,   # target angle
        30,   # degrees/sec
        1,    # direction: 1=CW, -1=CCW
        1,    # 0=absolute, 1=relative
        0, 0, 0)

    for i in range(40):
        msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1)
        if msg:
            print(f"    Heading: {msg.heading}  Alt: {msg.alt:.1f}m")
        time.sleep(0.5)
        if i > 6:
            break

    # Rotate 90 degrees left
    print("[*] Rotating 90 deg LEFT (CCW)...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        90,   # target angle
        30,   # degrees/sec
        -1,   # direction: CCW
        1,    # relative
        0, 0, 0)

    for i in range(40):
        msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1)
        if msg:
            print(f"    Heading: {msg.heading}  Alt: {msg.alt:.1f}m")
        time.sleep(0.5)
        if i > 6:
            break

    # Yaw rate test: send yaw rate via SET_POSITION_TARGET_LOCAL_NED
    print("[*] Yaw rate test: 0.3 rad/s for 3s...")
    TYPE_MASK = 0b010111000111  # ignore everything except yaw rate
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        master.mav.set_position_target_local_ned_send(
            0, master.target_system, master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            TYPE_MASK,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0.3)  # yaw=0, yaw_rate=0.3
        time.sleep(0.05)

    # Read heading after yaw rate
    msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=2)
    if msg:
        print(f"    Final heading: {msg.heading}  Alt: {msg.alt:.1f}m")

    # Stop rotation
    print("[*] Stopping rotation...")
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        TYPE_MASK,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.0)
    time.sleep(2)

    # Land
    print("[*] Landing...")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)  # LAND

    for _ in range(60):
        msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1)
        if msg:
            print(f"    Alt: {msg.alt:.1f}m")
            if msg.alt < 0.3:
                print("[+] Landed!")
                break
        time.sleep(0.5)

    # Disarm
    print("[*] Disarming...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0)

    print("\n[+] TEST COMPLETE!")
    master.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
