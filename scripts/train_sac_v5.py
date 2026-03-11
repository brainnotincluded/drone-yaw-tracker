#!/usr/bin/env python3
"""Train SAC model for drone yaw tracking — Loiter mode, PWM output, 50Hz.

Observation (100D): 10 timesteps × 10 features:
  [ang_err, yaw_rate, vy, vx, prev_action,
   err_rate, yaw_accel, lat_accel, err_integral, response_gain]
Action (1D):      [-1, 1] → PWM [1100-1900] on RC channel 4 (yaw)
Mode:             Loiter with RC override
Domain Rand:      Curriculum — none → light → full

v5.1 improvements:
  - 5-tier Gaussian reward with ultra-precision (σ=0.3°) for 0.5° target
  - DR curriculum: no DR first 20 eps, light 20-50, full 50+
  - Disturbance curriculum: no dist first 15 eps, then gradual increase
  - Smoothness/anti-osc penalties reduced near target to allow corrections
  - Best model saved by locked_avg (ignores initial search phase)
  - 600k timesteps

Usage:
    python train_sac_v5.py --timesteps 600000
    python train_sac_v5.py --resume models/sac_loiter.zip
"""

import sys
import os
import time
import math
import argparse
from collections import deque

import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.position_client import PositionClient
from src import drone


# Normalization constants
MAX_ANGLE = math.pi       # rad
MAX_YAW_RATE = 2.0        # rad/s
MAX_VEL = 5.0             # m/s
MAX_ERR_RATE = 3.0        # rad/s — angular error derivative
MAX_YAW_ACCEL = 10.0      # rad/s² — yaw angular acceleration
MAX_LAT_ACCEL = 5.0       # m/s² — lateral acceleration (disturbance proxy)
MAX_RESPONSE_GAIN = 5.0   # response_gain normalization
HISTORY = 10              # timesteps of history
STATE_DIM = 10            # per timestep
OBS_DIM = HISTORY * STATE_DIM  # 100

TARGET_ALT = 5.0          # m — training altitude


class DroneTrackEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, mav, pos_client, telemetry, rc_loop,
                 control_freq=50, episode_steps=1500, sim_speedup=1.0):
        super().__init__()
        self.mav = mav
        self.pos_client = pos_client
        self.telemetry = telemetry
        self.rc = rc_loop
        self.control_dt = 1.0 / control_freq / sim_speedup  # scale sleep for sim speed
        self.max_steps = episode_steps
        self.sim_speedup = sim_speedup

    def _sleep(self, secs):
        """Sleep scaled by sim speedup."""
        time.sleep(secs / self.sim_speedup)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.control_freq = control_freq
        self.step_count = 0
        self.prev_action = 0.0
        self.prev_prev_action = 0.0
        self.state_buffer = deque(maxlen=HISTORY)
        self.episode_errors = []

        # Derivative tracking
        self.prev_ang_err = 0.0
        self.prev_yaw_rate = 0.0
        self.prev_vy = 0.0
        self.error_history = deque(maxlen=HISTORY)  # for integral

        # Response gain (System ID feature)
        self.response_ema = 0.0
        self.current_err_rate = 0.0  # angular error derivative for reward

        # Domain Randomization params (randomized per episode in reset)
        self.response_scale = 1.0
        self.action_delay = 0
        self.sensor_noise_std = 0.0
        self.action_queue = deque()

        # Disturbance (RC roll override for lateral movement)
        self.dist_roll_pwm = 1500
        self.episode_number = 0
        self.is_dist_episode = False
        # Lock-on tracking
        self.locked_on = False
        self.locked_errors = []

    def _get_state(self):
        """Get current 10D state vector + position info."""
        pos = self.pos_client.get()
        telem = self.telemetry.get()

        if pos is None or pos.dist < 0.1:
            self.error_history.append(0.0)
            self.prev_ang_err = 0.0
            self.prev_yaw_rate = 0.0
            self.prev_vy = 0.0
            state = np.zeros(STATE_DIM, dtype=np.float32)
            return state, None

        ang_err = pos.relative_angle
        yaw_rate = telem['yaw_rate']
        vy = telem['vy']
        vx = telem['vx']

        # Domain Randomization: sensor noise
        if self.sensor_noise_std > 0:
            ang_err += np.random.normal(0, self.sensor_noise_std)
            yaw_rate += np.random.normal(0, self.sensor_noise_std * 0.5)
            vy += np.random.normal(0, self.sensor_noise_std * 0.3)
            vx += np.random.normal(0, self.sensor_noise_std * 0.3)

        # Derivatives (× control_freq to get per-second rates)
        err_rate = (ang_err - self.prev_ang_err) * self.control_freq
        yaw_accel = (yaw_rate - self.prev_yaw_rate) * self.control_freq
        lat_accel = (vy - self.prev_vy) * self.control_freq
        self.current_err_rate = err_rate  # store for reward function

        # Integral: rolling mean of recent angular errors
        self.error_history.append(ang_err)
        err_integral = float(np.mean(self.error_history))

        # Response gain: EMA of (delta_yaw_rate / delta_action)
        action_delta = self.prev_action - self.prev_prev_action
        yaw_rate_delta = yaw_rate - self.prev_yaw_rate
        if abs(action_delta) > 0.02:
            instant_gain = yaw_rate_delta / action_delta
            self.response_ema = 0.9 * self.response_ema + 0.1 * instant_gain

        # Update prev values for next step
        self.prev_ang_err = ang_err
        self.prev_yaw_rate = yaw_rate
        self.prev_vy = vy

        state = np.array([
            np.clip(ang_err / MAX_ANGLE, -1, 1),
            np.clip(yaw_rate / MAX_YAW_RATE, -1, 1),
            np.clip(vy / MAX_VEL, -1, 1),
            np.clip(vx / MAX_VEL, -1, 1),
            self.prev_action,
            np.clip(err_rate / MAX_ERR_RATE, -1, 1),
            np.clip(yaw_accel / MAX_YAW_ACCEL, -1, 1),
            np.clip(lat_accel / MAX_LAT_ACCEL, -1, 1),
            np.clip(err_integral / MAX_ANGLE, -1, 1),
            np.clip(self.response_ema / MAX_RESPONSE_GAIN, -1, 1),
        ], dtype=np.float32)

        return state, pos

    def _get_obs(self):
        """Get 100D observation (10 timesteps flattened)."""
        state, pos = self._get_state()
        self.state_buffer.append(state)
        obs = np.concatenate(list(self.state_buffer))
        return obs, pos

    def _reward(self, pos, action_val):
        if pos is None:
            return -2.0
        abs_err = abs(pos.relative_angle)
        deg = math.degrees(abs_err)

        # 5-tier Gaussian reward — heavily weighted toward precision
        r = 2.0 * math.exp(-(deg / 15.0) ** 2)    # Coarse: guide to general area
        r += 3.0 * math.exp(-(deg / 5.0) ** 2)     # Medium: get within 5°
        r += 5.0 * math.exp(-(deg / 2.0) ** 2)     # Fine: get within 2°
        r += 8.0 * math.exp(-(deg / 0.8) ** 2)     # Ultra: sub-degree precision
        r += 12.0 * math.exp(-(deg / 0.3) ** 2)    # Laser: 0.5° zone

        # Linear error penalty — strong push away from large errors
        r -= 2.0 * abs_err

        # === VELOCITY MATCHING near target ===
        # When close, reward low error rate (drone matching pedestrian motion)
        # This prevents overshoot oscillations that cause 2-5° average
        err_rate_deg = abs(math.degrees(self.current_err_rate))  # deg/s
        if deg < 5.0:
            # Big bonus for having low error rate when near target
            # err_rate_deg of 0 → +4.0, err_rate_deg of 10 → +0.5
            r += 4.0 * math.exp(-(err_rate_deg / 5.0) ** 2)
        if deg < 1.0:
            # Extra bonus for near-perfect stability
            r += 3.0 * math.exp(-(err_rate_deg / 2.0) ** 2)

        # Smoothness: moderate everywhere, lighter near target
        jerk = abs(action_val - self.prev_action)
        if deg > 5.0:
            r -= 0.25 * jerk
        else:
            r -= 0.10 * jerk  # still penalize jitter near target

        # Anti-oscillation: penalize reversals at all error levels
        delta_now = action_val - self.prev_action
        delta_prev = self.prev_action - self.prev_prev_action
        if delta_now * delta_prev < 0 and abs(delta_now) > 0.03:
            osc_mag = min(abs(delta_now), 0.1)
            if deg > 5.0:
                r -= 0.5 * osc_mag
            else:
                r -= 0.15 * osc_mag  # lighter but still present

        # Action magnitude — very light
        r -= 0.03 * action_val ** 2

        return r

    def _do_reset(self, telem_dbg):
        """Inner reset logic — may raise on MAVLink timeout."""
        alt = telem_dbg['alt']
        if alt < 1.5:
            print(f"  [!] Drone at alt={alt}m — re-arming and taking off", flush=True)
            drone.set_guided(self.mav)
            if not drone.arm(self.mav, timeout=15):
                raise RuntimeError("Re-arm failed")
            drone.takeoff(self.mav, 5.0)
            if not drone.wait_for_altitude(self.mav, 5.0, tolerance=0.5, timeout=30):
                raise RuntimeError("Failed to reach 5m")
            drone.set_loiter(self.mav)
            self._sleep(1)
        else:
            drone.set_guided(self.mav)
            self._sleep(0.3)
            drone.goto_home(self.mav, alt=5.0, timeout=40)

            pos_dbg2 = self.pos_client.get()
            if pos_dbg2:
                print(f"  [goto_home done] drone=({pos_dbg2.drone_x:.1f},{pos_dbg2.drone_y:.1f})", flush=True)

            drone.set_loiter(self.mav)
            self._sleep(0.3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Center roll/yaw
        self.rc.center_all()
        self._sleep(0.1)

        # Log position before reset for debugging
        telem_dbg = self.telemetry.get()
        pos_dbg = self.pos_client.get()
        if pos_dbg:
            print(f"  [reset] ep={self.episode_number} drone=({pos_dbg.drone_x:.1f},{pos_dbg.drone_y:.1f}) alt={telem_dbg['alt']:.1f}m", flush=True)

        # Robust reset with retry
        for attempt in range(3):
            try:
                self._do_reset(telem_dbg)
                break
            except Exception as e:
                print(f"  [!] Reset attempt {attempt+1} failed: {e}", flush=True)
                self.rc.center_all()
                self._sleep(2)
                if attempt == 2:
                    print("  [!] All reset attempts failed, forcing re-arm", flush=True)
                    drone.set_guided(self.mav)
                    self._sleep(1)
                    drone.arm(self.mav, timeout=15)
                    drone.takeoff(self.mav, 5.0)
                    drone.wait_for_altitude(self.mav, 5.0, tolerance=1.0, timeout=30)
                    drone.set_loiter(self.mav)
                    self._sleep(1)

        sys.stdout.flush()

        # Randomize initial yaw via RC loop (reduced from ±400 to ±150)
        # Smaller offset = less time wasted in "search" mode, more time tracking
        random_yaw_pwm = int(1500 + self.np_random.uniform(-150, 150))
        self.rc.set_yaw(random_yaw_pwm)
        self._sleep(0.3)
        self.rc.set_yaw(1500)
        self._sleep(0.2)

        # ===== Domain Randomization CURRICULUM =====
        # Phase 1 (ep 0-19): No DR — learn basic tracking on default dynamics
        # Phase 2 (ep 20-59): Light DR — gentle variation
        # Phase 3 (ep 60-119): Medium DR — wider range, still manageable
        # Phase 4 (ep 120+): Full DR — prepare for any drone
        ep = self.episode_number
        if ep < 20:
            self.response_scale = 1.0
            self.action_delay = 0
            self.sensor_noise_std = 0.0
        elif ep < 60:
            self.response_scale = self.np_random.uniform(0.85, 1.2)
            self.action_delay = int(self.np_random.integers(0, 2))  # 0-1 steps
            self.sensor_noise_std = self.np_random.uniform(0.0, 0.015)
        elif ep < 120:
            self.response_scale = self.np_random.uniform(0.7, 1.5)
            self.action_delay = int(self.np_random.integers(0, 3))  # 0-2 steps
            self.sensor_noise_std = self.np_random.uniform(0.0, 0.03)
        else:
            self.response_scale = self.np_random.uniform(0.5, 2.0)
            self.action_delay = int(self.np_random.integers(0, 5))  # 0-4 steps
            self.sensor_noise_std = self.np_random.uniform(0.0, 0.05)

        self.action_queue = deque([0.0] * (self.action_delay + 1),
                                  maxlen=self.action_delay + 1)
        print(f"  [DR] scale={self.response_scale:.2f} delay={self.action_delay} noise={self.sensor_noise_std:.3f}", flush=True)

        # Reset state
        self.step_count = 0
        self.prev_action = 0.0
        self.prev_prev_action = 0.0
        self.prev_ang_err = 0.0
        self.prev_yaw_rate = 0.0
        self.prev_vy = 0.0
        self.response_ema = 0.0
        self.current_err_rate = 0.0
        self.state_buffer.clear()
        self.error_history.clear()
        self.episode_errors = []
        self.locked_on = False
        self.locked_errors = []

        # Fill buffer with current state
        for _ in range(HISTORY):
            state, _ = self._get_state()
            self.state_buffer.append(state)
            time.sleep(self.control_dt)

        # ===== Disturbance CURRICULUM =====
        # First 15 episodes: no disturbance (focus on basic tracking)
        # Then: 5 no-dist + 6 dist per cycle, magnitude grows over time
        if ep < 20:
            self.is_dist_episode = False
            self.dist_roll_pwm = 1500
        else:
            cycle_pos = (ep - 20) % 11
            self.is_dist_episode = cycle_pos >= 5
            if self.is_dist_episode:
                # Disturbance magnitude grows slowly: start ±40, cap ±150
                max_offset = min(150, 40 + (ep - 20) * 1)
                offset = self.np_random.uniform(-max_offset, max_offset)
                self.dist_roll_pwm = int(1500 + offset)
            else:
                self.dist_roll_pwm = 1500

        self.episode_number += 1

        obs, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        a = float(np.clip(action[0], -1, 1))

        # Rate limit (0.2/step at 50Hz = full range in 10 steps = 200ms)
        max_delta = 0.20
        a = float(np.clip(a, self.prev_action - max_delta,
                           self.prev_action + max_delta))

        # Domain Randomization: action delay via FIFO queue
        self.action_queue.append(a)
        delayed_a = self.action_queue[0]

        # Domain Randomization: response scale
        scaled_a = delayed_a * self.response_scale

        # Convert to PWM
        yaw_pwm = int(1500 + scaled_a * 400)
        yaw_pwm = max(1100, min(1900, yaw_pwm))

        # Apply disturbance after settling (first 100 steps = 2s at 50Hz)
        roll_pwm = self.dist_roll_pwm if self.step_count > 100 else 1500

        self.rc.set(roll=roll_pwm, yaw=yaw_pwm)

        time.sleep(self.control_dt)

        obs, pos = self._get_obs()

        # Crash detection
        telem_now = self.telemetry.get()
        alt = telem_now.get('alt', 5.0)
        roll = telem_now.get('roll', 0.0)
        if alt < 1.0 or abs(roll) > 90:
            print(f"  [!] CRASH at step {self.step_count}: alt={alt:.1f}m roll={roll:.0f}°")
            self.rc.center_all()
            return obs, -10.0, False, True, {"crash": True}

        reward = self._reward(pos, a)

        if pos:
            err_deg = abs(math.degrees(pos.relative_angle))
            self.episode_errors.append(err_deg)
            if not self.locked_on and err_deg < 10.0:
                self.locked_on = True
            if self.locked_on:
                self.locked_errors.append(err_deg)

        self.prev_prev_action = self.prev_action
        self.prev_action = a
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        info = {}
        if truncated and self.episode_errors:
            info["avg_error"] = np.mean(self.episode_errors)
            info["pct_3deg"] = 100 * np.mean([e < 3 for e in self.episode_errors])
            info["pct_10deg"] = 100 * np.mean([e < 10 for e in self.episode_errors])
            info["pct_05deg"] = 100 * np.mean([e < 0.5 for e in self.episode_errors])
            info["pct_1deg"] = 100 * np.mean([e < 1 for e in self.episode_errors])
            info["dist_pwm"] = self.dist_roll_pwm
            info["is_dist"] = self.is_dist_episode
            info["dr_scale"] = self.response_scale
            info["dr_delay"] = self.action_delay
            if self.locked_errors:
                info["locked_avg"] = np.mean(self.locked_errors)
                info["locked_pct3"] = 100 * np.mean([e < 3 for e in self.locked_errors])
                info["locked_pct05"] = 100 * np.mean([e < 0.5 for e in self.locked_errors])

        return obs, reward, False, truncated, info


class TrackingCallback(BaseCallback):
    def __init__(self, save_path, log_path, save_freq=10, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.log_path = log_path
        self.save_freq = save_freq
        self.episode_count = 0
        self.best_error = float("inf")
        self.best_lock_error = float("inf")
        self.log_file = None

    def _on_training_start(self):
        self.log_file = open(self.log_path, "w")
        self.log_file.write("episode,timestep,avg_error,pct_05deg,pct_1deg,pct_3deg,pct_10deg,"
                            "locked_avg,locked_pct3,locked_pct05,dist_pwm,is_dist\n")
        self.log_file.flush()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "avg_error" in info:
                self.episode_count += 1
                avg_err = info["avg_error"]
                p05 = info.get("pct_05deg", 0)
                p1 = info.get("pct_1deg", 0)
                p3 = info["pct_3deg"]
                p10 = info["pct_10deg"]
                l_avg = info.get("locked_avg", -1)
                l_p3 = info.get("locked_pct3", -1)
                l_p05 = info.get("locked_pct05", -1)
                d_pwm = info.get("dist_pwm", 1500)
                is_dist = info.get("is_dist", False)

                self.log_file.write(
                    f"{self.episode_count},{self.num_timesteps},"
                    f"{avg_err:.2f},{p05:.1f},{p1:.1f},{p3:.1f},{p10:.1f},"
                    f"{l_avg:.2f},{l_p3:.1f},{l_p05:.1f},"
                    f"{d_pwm},{int(is_dist)}\n")
                self.log_file.flush()

                marker = ""
                # Save best model by locked_avg for no-dist episodes
                if not is_dist and l_avg >= 0 and l_avg < self.best_lock_error:
                    self.best_lock_error = l_avg
                    self.model.save(self.save_path + "_best")
                    marker = " *BEST*"
                elif avg_err < self.best_error:
                    self.best_error = avg_err

                if self.verbose:
                    lock_str = f"lock={l_avg:5.2f}°/<0.5°={l_p05:4.1f}%" if l_avg >= 0 else ""
                    dist_str = f"roll={d_pwm}" if is_dist else "no-dist"
                    print(f"  EP {self.episode_count:4d} | "
                          f"steps={self.num_timesteps:7d} | "
                          f"err={avg_err:5.2f}° | "
                          f"<0.5°={p05:4.1f}% | "
                          f"<1°={p1:4.1f}% | "
                          f"<3°={p3:4.1f}% | "
                          f"{dist_str} | "
                          f"{lock_str}{marker}")

                if self.episode_count % self.save_freq == 0:
                    self.model.save(self.save_path)
                    print(f"    -> Saved {self.save_path}")
        return True

    def _on_training_end(self):
        if self.log_file:
            self.log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Train SAC yaw tracker v5.1 (precision)")
    parser.add_argument("--timesteps", type=int, default=600000)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--control-freq", type=int, default=50)
    parser.add_argument("--altitude", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="rl_models_v5")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sim-speedup", type=float, default=1.0,
                        help="Sim speedup factor (match ArduCopter --speedup). Scales sleep time.")
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

    print("[*] Starting telemetry reader...")
    telemetry = drone.TelemetryReader(mav)
    time.sleep(0.5)

    print("[*] GUIDED + Arm + Takeoff...")
    drone.set_guided(mav)
    if not drone.arm(mav):
        print("[-] Arm failed")
        return 1
    drone.takeoff(mav, args.altitude)
    print(f"[*] Climbing to {args.altitude}m (waiting for actual altitude)...")
    if not drone.wait_for_altitude(mav, args.altitude, tolerance=0.5, timeout=30):
        print(f"[-] Failed to reach {args.altitude}m in 30s — aborting")
        return 1
    print("[+] At altitude")

    print("[*] Starting RC loop (50Hz continuous override)...")
    rc_loop = drone.RCLoop(mav, hz=args.control_freq)

    print("[*] Switching to Loiter...")
    drone.set_loiter(mav)
    time.sleep(1)

    ep_secs = args.episode_steps / args.control_freq
    ep_count = args.timesteps // args.episode_steps

    env = DroneTrackEnv(
        mav, pos_client, telemetry, rc_loop,
        control_freq=args.control_freq,
        episode_steps=args.episode_steps,
        sim_speedup=args.sim_speedup)

    model_path = os.path.join(args.save_dir, "sac_loiter")
    log_path = os.path.join(args.save_dir, "training_log.csv")

    if args.resume:
        print(f"[*] Resuming from {args.resume}")
        model = SAC.load(args.resume, env=env)
        # Speed up: 4x gradient steps per sim step + tighter entropy
        model.target_entropy = -2.0
        model.gradient_steps = 4
        model.batch_size = 256
        model.learning_starts = 1  # no warmup for resumed model
        print(f"[*] Set target_entropy={model.target_entropy} gradient_steps={model.gradient_steps}")
    else:
        model = SAC(
            "MlpPolicy", env,
            learning_rate=args.lr,
            buffer_size=300000,
            learning_starts=1500,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=4,  # 4x learning per sim step
            ent_coef="auto",
            target_entropy=-2.0,
            verbose=0,
            tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
            policy_kwargs=dict(net_arch=[256, 512, 256]),
        )

    callback = TrackingCallback(
        save_path=model_path, log_path=log_path, save_freq=10)

    print(f"\n{'=' * 60}")
    print(f"  SAC v5.1 TRAINING — PRECISION TARGET (0.5°)")
    print(f"  Timesteps:  {args.timesteps} (~{ep_count} episodes)")
    print(f"  Episode:    {args.episode_steps} steps ({ep_secs:.0f}s)")
    print(f"  Control:    {args.control_freq}Hz | PWM [1100-1900]")
    print(f"  Obs space:  {OBS_DIM}D ({HISTORY} × {STATE_DIM})")
    print(f"  Network:    MLP [256, 512, 256] | lr={args.lr}")
    print(f"  Buffer:     300k | batch=256 | grad_steps=4")
    print(f"  Rate limit: 0.20/step (50Hz) | sim_speedup={args.sim_speedup}x")
    print(f"  Reward:     5-tier Gaussian (σ=15,5,2,0.8,0.3)")
    print(f"  DR curriculum:")
    print(f"    EP 0-19:   No DR (learn basic tracking)")
    print(f"    EP 20-59:  Light DR (scale=[0.85,1.2] delay=[0,1])")
    print(f"    EP 60-119: Medium DR (scale=[0.7,1.5] delay=[0,2])")
    print(f"    EP 120+:   Full DR (scale=[0.5,2.0] delay=[0,4])")
    print(f"  Dist curriculum:")
    print(f"    EP 0-19:  No disturbance")
    print(f"    EP 20+:   5 no-dist + 6 dist/cycle, slow growth (±40→±150)")
    print(f"{'=' * 60}\n")

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(model_path)
        print(f"\n[+] Training complete! Model: {model_path}.zip")
    except KeyboardInterrupt:
        model.save(model_path)
        print(f"\n[!] Interrupted. Model saved: {model_path}.zip")
    except Exception as e:
        model.save(model_path)
        print(f"\n[!] CRASH: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print(f"[!] Model saved: {model_path}.zip")

    rc_loop.stop()
    print("[*] Landing...")
    drone.land(mav)
    time.sleep(10)
    telemetry.close()
    pos_client.close()
    mav.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
