"""Camera receiver for Webots drone camera stream (TCP, grayscale frames)."""

import struct
import socket
import numpy as np


class CameraReceiver:
    """Connects to the Webots camera TCP stream and reads grayscale frames.

    The drone controller sends frames as: [uint16 width][uint16 height][width*height bytes].
    """

    def __init__(self, host="127.0.0.1", port=5599, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))

    def get_frame(self):
        """Read one frame. Returns HxW uint8 numpy array or None."""
        try:
            hdr = self._recv_exact(4)
            if hdr is None:
                return None
            w, h = struct.unpack("<HH", hdr)
            data = self._recv_exact(w * h)
            if data is None:
                return None
            return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
        except Exception:
            return None

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 65536))
            if not chunk:
                return None
            buf += chunk
        return buf

    def close(self):
        if self.sock:
            self.sock.close()
