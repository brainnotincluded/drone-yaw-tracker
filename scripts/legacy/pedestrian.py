"""Randomized Pedestrian controller for Webots.

Drop-in replacement for the built-in pedestrian controller.
Adds trajectory randomization for RL drone-tracking training.

Usage:
  # Backward compatible (fixed trajectory):
  --trajectory "20 8, 27 1"

  # Random mode:
  --random --patterns straight,zigzag,circle,random_walk,figure8,evasive,stop_and_go
  --area "10 -5 35 15" --speed-range "0.3 1.5" --change-interval 20
  --reset-port 9998
"""
from controller import Supervisor

import optparse
import math
import random
import socket
import threading
import json


class RandomPedestrian(Supervisor):
    """Pedestrian with randomized movement trajectories."""

    def __init__(self):
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        self.speed = 0.5
        self.current_height_offset = 0
        self.joints_position_field = []
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]
        self.height_offsets = [
            -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03
        ]
        self.angles = [
            [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],
            [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
            [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],
            [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
            [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],
            [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
            [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],
            [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
            [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],
            [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
            [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],
            [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
            [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]
        ]
        Supervisor.__init__(self)

        # Randomization state
        self.random_mode = False
        self.patterns = []
        self.area = [10, -5, 35, 15]
        self.speed_range = [0.3, 1.5]
        self.change_interval = 20.0
        self.current_pattern = 'straight'
        self.paused = False
        self.pause_until = 0.0

        # Distance tracking (cumulative, survives trajectory changes)
        self.anim_distance = 0.0
        self.traj_distance = 0.0
        self.last_sim_time = 0.0

        # Reset control via TCP
        self._reset_requested = False
        self._reset_lock = threading.Lock()

    # ── Trajectory generators ────────────────────────────────────────────

    def _rand_point(self):
        """Random point inside the bounding area."""
        return [random.uniform(self.area[0], self.area[2]),
                random.uniform(self.area[1], self.area[3])]

    def _ensure_min_dist(self, p1, p2, min_d=3.0):
        """Regenerate p2 until it is at least min_d from p1."""
        for _ in range(50):
            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) >= min_d:
                return p2
            p2 = self._rand_point()
        return p2

    def generate_waypoints(self, pattern=None):
        if pattern is None:
            pattern = random.choice(self.patterns)
        self.current_pattern = pattern

        xmin, ymin, xmax, ymax = self.area
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        rx, ry = (xmax - xmin) / 2, (ymax - ymin) / 2

        gen = getattr(self, f'_gen_{pattern}', None)
        if gen:
            return gen(cx, cy, rx, ry)
        return self._gen_straight(cx, cy, rx, ry)

    def _gen_straight(self, cx, cy, rx, ry):
        p1 = self._rand_point()
        p2 = self._ensure_min_dist(p1, self._rand_point())
        return [p1, p2]

    def _gen_zigzag(self, cx, cy, rx, ry):
        n = random.randint(4, 8)
        xmin, ymin, xmax, ymax = self.area
        pts = []
        for i in range(n):
            x = xmin + (xmax - xmin) * i / (n - 1)
            y_off = random.uniform(2, max(2.1, ry)) * (1 if i % 2 == 0 else -1)
            pts.append([x, cy + y_off])
        return pts

    def _gen_circle(self, cx, cy, rx, ry):
        n = random.randint(8, 16)
        radius = random.uniform(3, max(3.1, min(rx, ry) * 0.8))
        ox = random.uniform(self.area[0] + radius, self.area[2] - radius)
        oy = random.uniform(self.area[1] + radius, self.area[3] - radius)
        return [[ox + radius * math.cos(2 * math.pi * i / n),
                 oy + radius * math.sin(2 * math.pi * i / n)] for i in range(n)]

    def _gen_random_walk(self, cx, cy, rx, ry):
        n = random.randint(6, 12)
        x, y = cx + random.uniform(-rx / 2, rx / 2), cy + random.uniform(-ry / 2, ry / 2)
        pts = []
        for _ in range(n):
            x = max(self.area[0], min(self.area[2], x + random.uniform(-5, 5)))
            y = max(self.area[1], min(self.area[3], y + random.uniform(-5, 5)))
            pts.append([x, y])
        return pts

    def _gen_figure8(self, cx, cy, rx, ry):
        n = 24
        r = random.uniform(3, max(3.1, min(rx, ry) * 0.4))
        return [[cx + r * math.sin(2 * math.pi * i / n),
                 cy + r * math.sin(4 * math.pi * i / n) / 2] for i in range(n)]

    def _gen_evasive(self, cx, cy, rx, ry):
        n = random.randint(8, 15)
        x, y = cx + random.uniform(-rx / 2, rx / 2), cy + random.uniform(-ry / 2, ry / 2)
        angle = random.uniform(0, 2 * math.pi)
        pts = []
        for _ in range(n):
            step = random.uniform(2, 6)
            angle += random.uniform(-math.pi * 0.8, math.pi * 0.8)
            x = max(self.area[0], min(self.area[2], x + step * math.cos(angle)))
            y = max(self.area[1], min(self.area[3], y + step * math.sin(angle)))
            pts.append([x, y])
        return pts

    def _gen_stop_and_go(self, cx, cy, rx, ry):
        return self._gen_straight(cx, cy, rx, ry)

    # ── Waypoint helpers ─────────────────────────────────────────────────

    def _set_waypoints(self, waypoints):
        self.waypoints = waypoints
        self.number_of_waypoints = len(waypoints)
        self.waypoints_distance = []
        for i in range(self.number_of_waypoints):
            j = (i + 1) % self.number_of_waypoints
            seg = math.hypot(waypoints[i][0] - waypoints[j][0],
                             waypoints[i][1] - waypoints[j][1])
            if i == 0:
                self.waypoints_distance.append(seg)
            else:
                self.waypoints_distance.append(self.waypoints_distance[-1] + seg)

    def _randomize_speed(self):
        self.speed = random.uniform(self.speed_range[0], self.speed_range[1])

    # ── Reset server ─────────────────────────────────────────────────────

    def _start_reset_server(self, port):
        def _serve():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(('0.0.0.0', port))
            srv.listen(2)
            srv.settimeout(1.0)
            print(f"[Pedestrian] Reset server listening on port {port}")
            while True:
                try:
                    conn, _ = srv.accept()
                    data = conn.recv(1024).decode().strip()
                    if data == 'reset':
                        with self._reset_lock:
                            self._reset_requested = True
                        conn.sendall(b'ok\n')
                    elif data == 'status':
                        info = json.dumps({
                            'pattern': self.current_pattern,
                            'speed': round(self.speed, 3),
                            'num_waypoints': self.number_of_waypoints,
                            'waypoints': [[round(w[0], 2), round(w[1], 2)] for w in self.waypoints],
                        })
                        conn.sendall((info + '\n').encode())
                    conn.close()
                except socket.timeout:
                    continue
                except Exception:
                    continue

        threading.Thread(target=_serve, daemon=True).start()

    # ── Walking animation ────────────────────────────────────────────────

    def _animate(self, is_paused):
        """Update joint angles for walking animation."""
        cycle = self.anim_distance / self.CYCLE_TO_DISTANCE_RATIO
        seq = int(cycle % self.WALK_SEQUENCES_NUMBER)
        ratio = cycle - int(cycle)

        for i in range(self.BODY_PARTS_NUMBER):
            if is_paused:
                self.joints_position_field[i].setSFFloat(0.0)
            else:
                a = self.angles[i][seq] * (1 - ratio) + \
                    self.angles[i][(seq + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                self.joints_position_field[i].setSFFloat(a)

        if is_paused:
            self.current_height_offset = 0
        else:
            self.current_height_offset = (
                self.height_offsets[seq] * (1 - ratio) +
                self.height_offsets[(seq + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
            )

    def _move(self):
        """Compute and set position along current trajectory."""
        total_loop = self.waypoints_distance[-1]
        if total_loop < 0.01:
            return
        rel = self.traj_distance - int(self.traj_distance / total_loop) * total_loop

        idx = 0
        for idx in range(self.number_of_waypoints):
            if self.waypoints_distance[idx] > rel:
                break

        if idx == 0:
            r = rel / self.waypoints_distance[0]
        else:
            r = (rel - self.waypoints_distance[idx - 1]) / \
                (self.waypoints_distance[idx] - self.waypoints_distance[idx - 1])

        j = (idx + 1) % self.number_of_waypoints
        x = r * self.waypoints[j][0] + (1 - r) * self.waypoints[idx][0]
        y = r * self.waypoints[j][1] + (1 - r) * self.waypoints[idx][1]

        angle = math.atan2(self.waypoints[j][1] - self.waypoints[idx][1],
                           self.waypoints[j][0] - self.waypoints[idx][0])

        self.root_translation_field.setSFVec3f([x, y, self.ROOT_HEIGHT + self.current_height_offset])
        self.root_rotation_field.setSFRotation([0, 0, 1, angle])

    # ── Main ─────────────────────────────────────────────────────────────

    def run(self):
        opt = optparse.OptionParser()
        opt.add_option("--trajectory", default="")
        opt.add_option("--speed", type=float, default=0.5)
        opt.add_option("--step", type=int, default=0)
        # Randomization
        opt.add_option("--random", action="store_true", default=False)
        opt.add_option("--patterns",
                       default="straight,zigzag,circle,random_walk,figure8,evasive,stop_and_go")
        opt.add_option("--area", default="10 -5 35 15")
        opt.add_option("--speed-range", default="0.3 1.5")
        opt.add_option("--change-interval", type=float, default=20.0)
        opt.add_option("--reset-port", type=int, default=0)
        opt.add_option("--seed", type=int, default=None)

        options, _ = opt.parse_args()

        self.time_step = options.step if options.step > 0 else int(self.getBasicTimeStep())

        # Get node fields
        self.root_node_ref = self.getSelf()
        self.root_translation_field = self.root_node_ref.getField("translation")
        self.root_rotation_field = self.root_node_ref.getField("rotation")
        for i in range(self.BODY_PARTS_NUMBER):
            self.joints_position_field.append(
                self.root_node_ref.getField(self.joint_names[i]))

        if options.random:
            self._init_random(options)
        else:
            self._init_fixed(options)

        # ── Simulation loop ──
        last_change = 0.0
        while self.step(self.time_step) != -1:
            sim_time = self.getTime()
            dt = sim_time - self.last_sim_time
            self.last_sim_time = sim_time

            # Handle trajectory changes in random mode
            if self.random_mode:
                do_change = False
                with self._reset_lock:
                    if self._reset_requested:
                        do_change = True
                        self._reset_requested = False
                if not do_change and sim_time - last_change >= self.change_interval:
                    do_change = True
                if do_change:
                    self._new_trajectory()
                    last_change = sim_time
                    print(f"[Pedestrian] T={sim_time:.0f}s: {self.current_pattern} "
                          f"speed={self.speed:.2f} waypoints={self.number_of_waypoints}")

                # stop_and_go pausing logic
                if self.current_pattern == 'stop_and_go':
                    if not self.paused and random.random() < 0.003:
                        self.paused = True
                        self.pause_until = sim_time + random.uniform(1.0, 4.0)
                    if self.paused and sim_time >= self.pause_until:
                        self.paused = False

                # evasive: occasional speed bursts
                if self.current_pattern == 'evasive' and random.random() < 0.005:
                    self.speed = random.uniform(self.speed_range[0], self.speed_range[1] * 1.3)

            # Advance distance
            if not self.paused:
                self.anim_distance += dt * self.speed
                self.traj_distance += dt * self.speed

            self._animate(self.paused)
            if not self.paused:
                self._move()

    def _init_random(self, options):
        self.random_mode = True
        if options.seed is not None:
            random.seed(options.seed)
        self.patterns = [p.strip() for p in options.patterns.split(',')]
        self.area = [float(x) for x in options.area.split()]
        self.speed_range = [float(x) for x in options.speed_range.split()]
        self.change_interval = options.change_interval

        self._new_trajectory()
        print(f"[Pedestrian] RANDOM mode enabled")
        print(f"[Pedestrian]   patterns:  {self.patterns}")
        print(f"[Pedestrian]   area:      {self.area}")
        print(f"[Pedestrian]   speed:     {self.speed_range}")
        print(f"[Pedestrian]   interval:  {self.change_interval}s")

        if options.reset_port > 0:
            self._start_reset_server(options.reset_port)

    def _init_fixed(self, options):
        self.speed = options.speed if options.speed > 0 else 0.5
        if not options.trajectory or len(options.trajectory.split(',')) < 2:
            print("You should specify the trajectory using the '--trajectory' option.")
            print("The trajectory should have at least 2 points.")
            return
        pts = []
        for part in options.trajectory.split(','):
            coords = part.strip().split()
            pts.append([float(coords[0]), float(coords[1])])
        self._set_waypoints(pts)

    def _new_trajectory(self):
        """Generate new random trajectory and reset distance."""
        wps = self.generate_waypoints()
        self._set_waypoints(wps)
        self._randomize_speed()
        self.traj_distance = 0.0
        self.paused = False


controller = RandomPedestrian()
controller.run()
