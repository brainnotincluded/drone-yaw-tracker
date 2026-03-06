"""HUD overlay drawing for tracking videos."""

import math
import cv2


def draw_crosshair(bgr):
    h, w = bgr.shape[:2]
    cv2.line(bgr, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 0), 1)
    cv2.line(bgr, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 0), 1)


def draw_target_indicator(bgr, rel_deg):
    """Draw red arrow showing where the human is relative to center."""
    h, w = bgr.shape[:2]
    px = int(w // 2 + rel_deg / 60.0 * (w // 2))
    px = max(10, min(w - 10, px))
    cv2.arrowedLine(bgr, (px, 30), (px, 60), (0, 0, 255), 2)
    cv2.circle(bgr, (px, 30), 6, (0, 0, 255), -1)


def draw_info(bgr, lines):
    """Draw text lines in bottom-left corner."""
    h, w = bgr.shape[:2]
    for i, txt in enumerate(lines):
        cv2.putText(bgr, txt, (10, h - 15 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


def draw_disturbance(bgr, vy):
    """Draw disturbance indicator with red border."""
    h, w = bgr.shape[:2]
    cv2.putText(bgr, f"ROLL DIST: vy={vy:+.1f} m/s", (w // 2 - 130, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.rectangle(bgr, (5, 5), (w - 5, h - 5), (0, 0, 255), 3)


def draw_tracking_hud(bgr, pos, action, yaw_rate_cmd, elapsed, label="RL",
                      dist_vy=0.0, dist_active=False):
    """Full tracking HUD: crosshair + indicator + info + optional disturbance."""
    draw_crosshair(bgr)

    if pos:
        rel_deg = math.degrees(pos.relative_angle)
        draw_target_indicator(bgr, rel_deg)
        draw_info(bgr, [
            f"dist={pos.dist:.1f}m",
            f"rel={rel_deg:+.1f} deg",
            f"action={action:+.2f}",
            f"yaw_cmd={math.degrees(yaw_rate_cmd):+.1f} deg/s",
            f"T={elapsed:.1f}s  [{label}]",
        ])

    if dist_active:
        draw_disturbance(bgr, dist_vy)
