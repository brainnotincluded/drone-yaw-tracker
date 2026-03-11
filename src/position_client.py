"""
Client for Webots position server (port 9999).

Usage:
    from position_client import PositionClient

    client = PositionClient()
    client.connect()

    pos = client.get()
    print(pos.drone_x, pos.drone_y, pos.drone_yaw)
    print(pos.human_x, pos.human_y)
    print(pos.dx, pos.dy, pos.dist, pos.angle)

    # Direction relative to drone heading:
    print(pos.relative_angle)  # >0 = human is to the right

    client.close()
"""

import socket
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Positions:
    """Snapshot of drone/human positions and direction vector."""
    # Drone (Webots world frame)
    drone_x: float = 0.0
    drone_y: float = 0.0
    drone_z: float = 0.0
    drone_yaw: float = 0.0  # radians

    # Human (Webots world frame)
    human_x: float = 0.0
    human_y: float = 0.0
    human_z: float = 0.0

    # Direction vector (world frame)
    dx: float = 0.0
    dy: float = 0.0
    dist: float = 0.0
    angle: float = 0.0  # bearing drone→human, radians

    @property
    def relative_angle(self) -> float:
        """Angle from drone heading to human, in [-pi, pi].
        Positive = human is to the right of where drone is facing."""
        diff = self.angle - self.drone_yaw
        return (diff + math.pi) % (2 * math.pi) - math.pi


class PositionClient:
    """Connects to Webots supervisor position stream."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._buf = ""
        self._latest: Optional[Positions] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self, background: bool = True) -> bool:
        """Connect to position server.

        Args:
            background: if True, read in a background thread (non-blocking get()).
                        if False, get() will read from socket directly.
        """
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

    def _parse(self, line: str) -> Positions:
        d = json.loads(line)
        return Positions(
            drone_x=d["drone"]["x"], drone_y=d["drone"]["y"],
            drone_z=d["drone"]["z"], drone_yaw=d["drone"]["yaw"],
            human_x=d["human"]["x"], human_y=d["human"]["y"],
            human_z=d["human"]["z"],
            dx=d["vec"]["dx"], dy=d["vec"]["dy"],
            dist=d["vec"]["dist"], angle=d["vec"]["angle"],
        )

    def get(self, timeout: float = 1.0) -> Optional[Positions]:
        """Get latest position snapshot.

        In background mode: returns cached latest (non-blocking).
        In foreground mode: reads one message from socket.
        """
        if self._thread:
            with self._lock:
                return self._latest

        # Foreground: read directly
        self._sock.settimeout(timeout)
        try:
            data = self._sock.recv(4096).decode()
            self._buf += data
            lines = self._buf.split("\n")
            self._buf = lines[-1]
            for line in reversed(lines[:-1]):
                if line.strip():
                    return self._parse(line)
        except socket.timeout:
            pass
        return self._latest

    def close(self):
        self._running = False
        if self._sock:
            self._sock.close()


# ── Quick test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = PositionClient()
    if not client.connect():
        exit(1)

    time.sleep(0.2)
    for i in range(20):
        p = client.get()
        if p:
            print(f"Drone: ({p.drone_x:7.2f}, {p.drone_y:7.2f}, {p.drone_z:5.2f})  "
                  f"yaw={math.degrees(p.drone_yaw):6.1f} deg")
            print(f"Human: ({p.human_x:7.2f}, {p.human_y:7.2f}, {p.human_z:5.2f})")
            print(f"Vector: dist={p.dist:6.2f}m  "
                  f"bearing={math.degrees(p.angle):6.1f} deg  "
                  f"relative={math.degrees(p.relative_angle):+6.1f} deg")
            print()
        time.sleep(0.5)

    client.close()
