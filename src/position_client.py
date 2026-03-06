"""Client for Webots supervisor position stream (TCP port 9999).

Usage:
    from src.position_client import PositionClient

    client = PositionClient()
    client.connect()
    pos = client.get()
    print(pos.relative_angle)  # radians, >0 = human to the right
    client.close()
"""

import socket
import json
import math
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class Positions:
    """Snapshot of drone/human positions."""
    drone_x: float = 0.0
    drone_y: float = 0.0
    drone_z: float = 0.0
    drone_yaw: float = 0.0

    human_x: float = 0.0
    human_y: float = 0.0
    human_z: float = 0.0

    dx: float = 0.0
    dy: float = 0.0
    dist: float = 0.0
    angle: float = 0.0  # bearing drone→human (radians, world frame)

    @property
    def relative_angle(self) -> float:
        """Angle from drone heading to human, in [-pi, pi].
        Positive = human is to the right of drone heading."""
        diff = self.angle - self.drone_yaw
        return (diff + math.pi) % (2 * math.pi) - math.pi


class PositionClient:
    """Connects to Webots supervisor position stream."""

    def __init__(self, host="127.0.0.1", port=9999):
        self.host = host
        self.port = port
        self._sock = None
        self._latest = None
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

    def connect(self, background=True):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        try:
            self._sock.connect((self.host, self.port))
        except socket.error as e:
            print(f"[PositionClient] Connection failed: {e}")
            return False
        if background:
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        return True

    def _read_loop(self):
        buf = ""
        while self._running:
            try:
                data = self._sock.recv(4096).decode()
                if not data:
                    break
                buf += data
                lines = buf.split("\n")
                buf = lines[-1]
                for line in lines[:-1]:
                    if line.strip():
                        pos = self._parse(line)
                        with self._lock:
                            self._latest = pos
            except socket.timeout:
                continue
            except Exception:
                break

    def _parse(self, line):
        d = json.loads(line)
        return Positions(
            drone_x=d["drone"]["x"], drone_y=d["drone"]["y"],
            drone_z=d["drone"]["z"], drone_yaw=d["drone"]["yaw"],
            human_x=d["human"]["x"], human_y=d["human"]["y"],
            human_z=d["human"]["z"],
            dx=d["vec"]["dx"], dy=d["vec"]["dy"],
            dist=d["vec"]["dist"], angle=d["vec"]["angle"],
        )

    def get(self, timeout=1.0) -> Optional[Positions]:
        if self._thread:
            with self._lock:
                return self._latest
        self._sock.settimeout(timeout)
        try:
            data = self._sock.recv(4096).decode()
            lines = data.split("\n")
            for line in reversed(lines):
                if line.strip():
                    return self._parse(line)
        except socket.timeout:
            pass
        return self._latest

    def close(self):
        self._running = False
        if self._sock:
            self._sock.close()
