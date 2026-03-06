#!/usr/bin/env python3
"""Train PPO model for drone yaw tracking.

Observation (3D): [relative_angle_norm, angular_velocity_norm, prev_action]
Action (1D):      yaw rate [-1, 1] scaled to max_yaw_rate
Reward:           -|rel_angle| + bonuses for tight tracking - jerk penalty

Usage:
    python scripts/train.py --timesteps 50000
    python scripts/train.py --resume models/ppo_tracker.zip
"""

import sys
import os
import time
import math
import socket
import argparse

import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.position_client import PositionClient
from src import drone


class DroneTrackEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, mav, pos_client, max_yaw_rate=1.5,
                 control_freq=10, episode_steps=300):
        super().__init__()
        self.mav = mav
        self.pos_client = pos_client
        self.max_yaw_rate = max_yaw_rate
        self.control_dt = 1.0 / control_freq
        self.max_steps = episode_steps

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.step_count = 0
        self.prev_action = 0.0
        self.prev_rel_angle = 0.0
        self.episode_errors = []

    def _get_obs(self):
        pos = self.pos_client.get()
        if pos is None or pos.dist < 0.1:
            return np.zeros(3, dtype=np.float32), None

        rel = pos.relative_angle
        diff = rel - self.prev_rel_angle
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        ang_vel = diff / max(self.control_dt, 0.001)

        obs = np.array([
            np.clip(rel / math.pi, -1, 1),
            np.clip(ang_vel / 2.0, -1, 1),
            self.prev_action
        ], dtype=np.float32)

        self.prev_rel_angle = rel
        return obs, pos

    def _reward(self, pos, action_val):
        if pos is None:
            return -1.0
        abs_err = abs(pos.relative_angle)
        deg = math.degrees(abs_err)
        r = -abs_err
        if deg < 3:
            r += 1.0
        elif deg < 10:
            r += 0.3
        elif deg < 20:
            r += 0.1
        r -= 0.05 * abs(action_val - self.prev_action)
        return r

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(("127.0.0.1", 9998))
            s.sendall(b"reset\n")
            s.close()
        except Exception:
            pass
        drone.send_yaw_rate(self.mav, 0)
        time.sleep(0.3)
        self.step_count = 0
        self.prev_action = 0.0
        self.prev_rel_angle = 0.0
        self.episode_errors = []
        obs, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        a = float(np.clip(action[0], -1, 1))
        yaw_rate = -a * self.max_yaw_rate
        drone.send_yaw_rate(self.mav, yaw_rate)
        time.sleep(self.control_dt)
        obs, pos = self._get_obs()
        reward = self._reward(pos, a)
        if pos:
            self.episode_errors.append(abs(math.degrees(pos.relative_angle)))
        self.prev_action = a
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        info = {}
        if truncated and self.episode_errors:
            info["avg_error"] = np.mean(self.episode_errors)
            info["pct_3deg"] = 100 * np.mean([e < 3 for e in self.episode_errors])
            info["pct_10deg"] = 100 * np.mean([e < 10 for e in self.episode_errors])
        return obs, reward, False, truncated, info


class TrackingCallback(BaseCallback):
    def __init__(self, save_path, log_path, save_freq=10, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.log_path = log_path
        self.save_freq = save_freq
        self.episode_count = 0
        self.best_error = float("inf")
        self.log_file = None

    def _on_training_start(self):
        self.log_file = open(self.log_path, "w")
        self.log_file.write("episode,timestep,avg_error,pct_3deg,pct_10deg\n")
        self.log_file.flush()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "avg_error" in info:
                self.episode_count += 1
                avg_err = info["avg_error"]
                p3 = info["pct_3deg"]
                p10 = info["pct_10deg"]
                self.log_file.write(
                    f"{self.episode_count},{self.num_timesteps},"
                    f"{avg_err:.1f},{p3:.1f},{p10:.1f}\n")
                self.log_file.flush()
                if self.verbose:
                    marker = ""
                    if avg_err < self.best_error:
                        self.best_error = avg_err
                        marker = " *BEST*"
                    print(f"  EP {self.episode_count:4d} | "
                          f"steps={self.num_timesteps:7d} | "
                          f"err={avg_err:5.1f}° | "
                          f"<3°={p3:4.1f}% | "
                          f"<10°={p10:4.1f}%{marker}")
                if self.episode_count % self.save_freq == 0:
                    self.model.save(self.save_path)
                    print(f"    -> Saved {self.save_path}")
        return True

    def _on_training_end(self):
        if self.log_file:
            self.log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO yaw tracker")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--max-yaw-rate", type=float, default=1.5)
    parser.add_argument("--altitude", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("[*] Connecting position client...")
    pos_client = PositionClient()
    if not pos_client.connect():
        return 1
    time.sleep(0.3)

    print("[*] Connecting drone...")
    mav = drone.connect()
    print(f"[+] Drone system {mav.target_system}")

    print("[*] GUIDED + Arm + Takeoff...")
    drone.set_guided(mav)
    if not drone.arm(mav):
        print("[-] Arm failed")
        return 1
    drone.takeoff(mav, args.altitude)
    print(f"[*] Climbing to {args.altitude}m...")
    time.sleep(15)
    print("[+] At altitude")

    env = DroneTrackEnv(
        mav, pos_client,
        max_yaw_rate=args.max_yaw_rate,
        control_freq=args.control_freq,
        episode_steps=args.episode_steps)

    model_path = os.path.join(args.save_dir, "ppo_tracker")
    log_path = os.path.join(args.save_dir, "training_log.csv")

    if args.resume:
        print(f"[*] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=args.lr,
            n_steps=args.episode_steps,
            batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            verbose=0,
            tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
            policy_kwargs=dict(net_arch=[64, 64]),
        )

    callback = TrackingCallback(
        save_path=model_path, log_path=log_path, save_freq=10)

    ep_count = args.timesteps // args.episode_steps
    ep_secs = args.episode_steps / args.control_freq

    print(f"\n{'=' * 60}")
    print(f"  RL TRAINING (PPO)")
    print(f"  Timesteps:  {args.timesteps} ({ep_count} episodes)")
    print(f"  Episode:    {args.episode_steps} steps ({ep_secs:.0f}s realtime)")
    print(f"  Control:    {args.control_freq}Hz | max yaw: {args.max_yaw_rate} rad/s")
    print(f"  Network:    MLP [64, 64] | lr={args.lr}")
    print(f"{'=' * 60}\n")

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(model_path)
        print(f"\n[+] Training complete! Model: {model_path}.zip")
    except KeyboardInterrupt:
        model.save(model_path)
        print(f"\n[!] Interrupted. Model saved: {model_path}.zip")

    drone.send_yaw_rate(mav, 0)
    print("[*] Landing...")
    drone.land(mav)
    time.sleep(10)
    pos_client.close()
    mav.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
