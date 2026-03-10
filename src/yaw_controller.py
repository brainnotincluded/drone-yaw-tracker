"""Drop-in yaw tracking controller. PWM output, 50Hz.

Usage:
    from yaw_controller import YawController

    ctrl = YawController("sac_loiter.zip")

    # In your 50Hz loop:
    pwm = ctrl.predict(angular_error, yaw_rate, vy, vx)
    # Send pwm to RC channel 4
"""

import numpy as np
from collections import deque
from stable_baselines3 import SAC


HISTORY = 10
STATE_DIM = 5
MAX_ANGLE = 3.14159265
MAX_YAW_RATE = 2.0
MAX_VEL = 5.0


class YawController:
    def __init__(self, model_path, max_delta=0.12):
        """Load model and initialize state buffer.

        Args:
            model_path: Path to SAC .zip model file.
            max_delta: Action rate limit per step (0.12 at 50Hz).
        """
        self.model = SAC.load(model_path)
        self.max_delta = max_delta
        self.prev_action = 0.0
        self.buffer = deque(maxlen=HISTORY)

        # Fill buffer with zeros
        for _ in range(HISTORY):
            self.buffer.append(np.zeros(STATE_DIM, dtype=np.float32))

    def predict(self, angular_error, yaw_rate, vy, vx):
        """Get yaw PWM command.

        Call this at 50Hz. Maintains internal state buffer.

        Args:
            angular_error: Angle to target in radians. Positive = target left.
            yaw_rate: Current yaw rate in rad/s.
            vy: Lateral velocity in m/s.
            vx: Longitudinal velocity in m/s.

        Returns:
            int: PWM value [1100-1900]. 1500 = no rotation.
        """
        state = np.array([
            np.clip(angular_error / MAX_ANGLE, -1, 1),
            np.clip(yaw_rate / MAX_YAW_RATE, -1, 1),
            np.clip(vy / MAX_VEL, -1, 1),
            np.clip(vx / MAX_VEL, -1, 1),
            self.prev_action,
        ], dtype=np.float32)

        self.buffer.append(state)

        obs = np.concatenate(list(self.buffer))
        action, _ = self.model.predict(obs, deterministic=True)
        a = float(np.clip(action[0], -1, 1))

        # Rate limit
        a = float(np.clip(a, self.prev_action - self.max_delta,
                           self.prev_action + self.max_delta))
        self.prev_action = a

        pwm = int(1500 + a * 400)
        return max(1100, min(1900, pwm))

    def reset(self):
        """Reset state buffer. Call between episodes or target switches."""
        self.buffer.clear()
        for _ in range(HISTORY):
            self.buffer.append(np.zeros(STATE_DIM, dtype=np.float32))
        self.prev_action = 0.0
