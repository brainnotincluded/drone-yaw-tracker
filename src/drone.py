"""MAVLink helpers for ArduPilot SITL drone control."""

import time
from pymavlink import mavutil

# Ignore position, use velocity, ignore acceleration, ignore yaw, use yaw_rate
TYPE_MASK_YAW_RATE = 0b010111000111


def connect(conn_str="udp:127.0.0.1:14550"):
    """Connect to drone via MAVLink, wait for heartbeat, request data streams."""
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


def arm(m, timeout=30):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    for _ in range(timeout):
        msg = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            return True
    return False


def takeoff(m, alt=5.0):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt)


def send_yaw_rate(m, yaw_rate, vy=0.0):
    """Send yaw rate command with optional lateral velocity (body frame).

    Args:
        yaw_rate: Yaw rate in rad/s.
        vy: Lateral velocity in m/s (body frame). Positive = right. Causes roll.
    """
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        TYPE_MASK_YAW_RATE,
        0, 0, 0,
        0, float(vy), 0,
        0, 0, 0,
        0, float(yaw_rate))


def land(m):
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
