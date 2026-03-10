"""MAVLink helpers for ArduPilot SITL drone control."""

import math
import time
import threading
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


def set_sitl_params(m):
    """Set SITL-specific parameters for stable Loiter flight."""
    params = [
        (b"ARMING_CHECK", 0.0),       # skip prearm checks in SITL
        (b"MOT_THST_HOVER", 0.31),    # correct hover thrust for Webots Iris
        (b"MOT_HOVER_LEARN", 0.0),    # disable learning (our RC doesn't confuse it)
    ]
    for name, val in params:
        m.mav.param_set_send(m.target_system, m.target_component, name, val, 9)
        time.sleep(0.3)


def arm(m, timeout=30):
    set_sitl_params(m)  # ensure SITL params are set before arming
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


def send_yaw_rate(m, yaw_rate, vy=0.0, drone_yaw=None):
    """Send yaw rate command with optional lateral velocity.

    When vy is non-zero, uses LOCAL_NED frame with world-frame velocity
    computed from the drone's current heading. This avoids the ArduPilot bug
    where BODY_NED vy overrides yaw_rate control.

    Args:
        yaw_rate: Yaw rate in rad/s.
        vy: Lateral velocity in m/s (body right). Positive = right. Causes roll.
        drone_yaw: Current drone yaw in radians (required when vy != 0).
    """
    if vy == 0.0 or drone_yaw is None:
        # Simple case: yaw rate only, body frame
        m.mav.set_position_target_local_ned_send(
            0, m.target_system, m.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            TYPE_MASK_YAW_RATE,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, float(yaw_rate))
    else:
        # Convert body-right vy to world-frame NED velocity
        # Body right vector in NED: [sin(yaw), cos(yaw)]
        # (if yaw=0 facing north, right=east: vx_ned=0, vy_ned=vy ✓)
        vx_ned = -float(vy) * math.sin(drone_yaw)
        vy_ned = float(vy) * math.cos(drone_yaw)
        m.mav.set_position_target_local_ned_send(
            0, m.target_system, m.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            TYPE_MASK_YAW_RATE,
            0, 0, 0,
            vx_ned, vy_ned, 0,
            0, 0, 0,
            0, float(yaw_rate))


def get_altitude(m, timeout=3):
    """Return current relative altitude in metres (positive = up). None on timeout."""
    msg = m.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=timeout)
    return -msg.z if msg else None


def get_roll_deg(m, timeout=3):
    """Return current roll angle in degrees. None on timeout."""
    msg = m.recv_match(type='ATTITUDE', blocking=True, timeout=timeout)
    return math.degrees(msg.roll) if msg else None


def is_armed(m, timeout=3):
    """Return True if drone is currently armed."""
    msg = m.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
    return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) if msg else False


def wait_for_altitude(m, target_alt, tolerance=0.5, timeout=30):
    """Block until drone reaches target altitude (within tolerance). Returns True/False."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        alt = get_altitude(m, timeout=2)
        if alt is not None and alt >= target_alt - tolerance:
            return True
        time.sleep(0.2)
    return False


def goto_home(m, alt=5.0, timeout=15):
    """Fly back to local NED origin (0,0) at given altitude."""
    # Position mask: use position, ignore velocity/accel, ignore yaw/yaw_rate
    MASK_POS = 0b110111111000
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        MASK_POS,
        0, 0, -alt,  # NED: x=0, y=0, z=-alt (up)
        0, 0, 0,
        0, 0, 0,
        0, 0)
    time.sleep(timeout)


def reset_position(host="127.0.0.1", port=9997, x=0.0, y=0.0, z=None):
    """Instantly teleport drone via Webots supervisor (port 9997).

    Takes ~50ms instead of 5-15s flying.
    """
    import socket as _socket
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.settimeout(5.0)
    try:
        s.connect((host, port))
        if z is not None:
            s.sendall(f"reset {x} {y} {z}\n".encode())
        else:
            s.sendall(b"reset\n")
        resp = s.recv(64).decode().strip()
        s.close()
        return resp == "ok"
    except Exception as e:
        s.close()
        print(f"[drone] reset_position failed: {e}")
        return False


def set_loiter(m):
    """Switch to Loiter mode (mode 5)."""
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 5)
    time.sleep(0.5)


def send_rc_override(m, roll=0, pitch=0, throttle=0, yaw=0):
    """Send RC channel override. 0 = release (no override).
    PWM range: 1100-1900, center 1500.
    """
    m.mav.rc_channels_override_send(
        m.target_system, m.target_component,
        roll, pitch, throttle, yaw,
        0, 0, 0, 0)


def send_yaw_pwm(m, pwm):
    """Send RC override for yaw channel only (channel 4)."""
    pwm = max(1100, min(1900, int(pwm)))
    send_rc_override(m, yaw=pwm)


def release_rc(m):
    """Release all RC overrides."""
    send_rc_override(m)


class RCLoop:
    """Background thread that continuously sends RC override at 50Hz.

    RC override must be sent continuously like a real transmitter.
    To change a channel, update the value in the channels array.

    Usage:
        rc = RCLoop(mav)          # starts pumping [1500,1500,1500,1500]
        rc.set_yaw(1700)          # change yaw
        rc.set_roll(1600)         # change roll
        rc.center_all()           # back to 1500 on all channels
        rc.stop()                 # stop and release
    """

    def __init__(self, mav, hz=50):
        self.mav = mav
        self.dt = 1.0 / hz
        # [roll, pitch, throttle, yaw]
        # throttle=1500 = hold altitude in Loiter (SITL has no hardware RC, so 65535 defaults to min throttle)
        self._ch = [1500, 1500, 1500, 1500]
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            with self._lock:
                r, p, t, y = self._ch
            self.mav.mav.rc_channels_override_send(
                self.mav.target_system, self.mav.target_component,
                r, p, t, y, 0, 0, 0, 0)
            time.sleep(self.dt)

    def set_roll(self, pwm):
        with self._lock:
            self._ch[0] = max(1100, min(1900, int(pwm)))

    def set_pitch(self, pwm):
        with self._lock:
            self._ch[1] = max(1100, min(1900, int(pwm)))

    def set_throttle(self, pwm):
        with self._lock:
            self._ch[2] = max(1100, min(1900, int(pwm)))

    def set_yaw(self, pwm):
        with self._lock:
            self._ch[3] = max(1100, min(1900, int(pwm)))

    def set(self, roll=None, pitch=None, throttle=None, yaw=None):
        with self._lock:
            if roll is not None:
                self._ch[0] = max(1100, min(1900, int(roll)))
            if pitch is not None:
                self._ch[1] = max(1100, min(1900, int(pitch)))
            if throttle is not None:
                self._ch[2] = max(1100, min(1900, int(throttle)))
            if yaw is not None:
                self._ch[3] = max(1100, min(1900, int(yaw)))

    def center_all(self):
        with self._lock:
            self._ch[:] = [1500, 1500, 1500, 1500]

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
        # Release all overrides
        send_rc_override(self.mav)


class TelemetryReader:
    """Background reader for MAVLink telemetry (yaw_rate, vx, vy, alt, roll)."""

    def __init__(self, mav):
        self.mav = mav
        self._data = {'yaw_rate': 0.0, 'vx': 0.0, 'vy': 0.0, 'alt': 0.0, 'roll': 0.0}
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                msg = self.mav.recv_msg()
                if msg is None:
                    time.sleep(0.001)
                    continue
                t = msg.get_type()
                with self._lock:
                    if t == 'ATTITUDE':
                        self._data['yaw_rate'] = msg.yawspeed
                        self._data['roll'] = math.degrees(msg.roll)
                    elif t == 'LOCAL_POSITION_NED':
                        self._data['vx'] = msg.vx
                        self._data['vy'] = msg.vy
                        self._data['alt'] = -msg.z  # NED z is negative up
            except Exception:
                time.sleep(0.01)

    def get(self):
        with self._lock:
            return dict(self._data)

    def close(self):
        self._running = False


def land(m):
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
