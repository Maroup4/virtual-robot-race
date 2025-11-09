# rule_based_algorithms/driver_model.py
# ----------------------------------------------------------------------
# STEER-TYPE CONTROL VERSION
# Higher layer decides lane_mode ("normal" / "hold" / "search") and lost_age.
# This driver computes: (drive_torque, steer_angle) for steer-type robots.
#
# Sign conventions:
#   lateral_px : right is + [px] (offset from image center)
#   theta_deg  : 0° is up, tilting to the right is + [deg]
#   Output:
#     drive_torque : forward torque (-1.0 to +1.0, + is forward)
#     steer_angle  : steering angle in radians (+ is right turn)
# ----------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math

def _clip(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


@dataclass
class DriverConfig:
    # Geometry / normalization
    image_width: int = 224
    lateral_norm_halfwidth_px: Optional[float] = None  # None → image_width/2

    # Speed planning (auto slow-down in curves)
    v_min: float = 0.20
    v_max: float = 0.70
    slow_w_theta: float = 0.70
    slow_w_lateral: float = 0.60

    # Steering blend (Yaw: right is +)
    k_theta: float = 0.90          # Gain for theta (angle)
    k_lateral: float = 0.60        # Gain for lateral offset
    steer_limit: float = 0.785     # Max steer angle (radians, ~45 deg)

    # Output shaping
    torque_limit: float = 1.00
    alpha_smooth: float = 0.30     # 0: off, 1: heavy smoothing (IIR)

    # Battery (SOC) scaling
    use_soc_scaling: bool = True
    soc_floor: float = 0.30

    # Conventions / safety
    forward_sign: int = +1
    theta_hard_limit_deg: float = 80.0
    invalid_brake: float = 0.0        # output when NO-GO etc. (0.0 recommended)
    require_start_go: bool = True     # True: must stop if NO-GO

    # ====== Lane-lost related (pure math parameters; mode decision is upstream) ======
    hold_decay_per_frame: float = 0.90   # decay ratio during hold
    search_pivot: bool = True            # True: pivot in place / False: slow forward while turning
    search_speed: float = 0.00           # forward component (0.0 when pivoting)
    search_steer_const: float = 0.6      # [rad] constant steer angle during search (~34 deg)
    loop_period_s: float = 0.050         # for logs only


class DriverModel:
    """Given lane_mode, produce (drive_torque, steer_angle) for steer-type control."""

    def __init__(self, cfg: DriverConfig):
        self.cfg = cfg
        self._prev_drive = 0.0
        self._prev_steer = 0.0
        self._half_w = max(
            1.0,
            (cfg.lateral_norm_halfwidth_px
             if cfg.lateral_norm_halfwidth_px is not None
             else cfg.image_width * 0.5)
        )
        # For hold mode, remember most recent base/steer
        self._last_base = 0.0
        self._last_steer = 0.0
        self.last_debug: Dict[str, float | int | str | bool | None] = {}

    def update(
        self,
        lateral_px: Optional[float],
        theta_deg: Optional[float],
        soc: Optional[float],
        image_width: Optional[int],
        start_go: bool,
        valid_lane: bool,
        lane_mode: str = "normal",   # "normal" / "hold" / "search"
        lost_age: int = 0,           # consecutive lost-frame count (for info)
    ) -> Tuple[float, float]:
        """Compute (drive_torque, steer_angle) for the current frame."""
        # Geometry update
        if image_width is not None and image_width > 0:
            self._half_w = max(1.0, image_width * 0.5)

        # Start gate
        if self.cfg.require_start_go and not start_go:
            drive = steer = self.cfg.invalid_brake * self.cfg.forward_sign
            drive, steer = self._smooth(drive, steer)
            self._store_debug(False, False, lane_mode, lateral_px, theta_deg,
                              0.0, 0.0, 0.0, drive, steer, soc, None, None, lost_age)
            return drive, steer

        # Angle sanity: if theta is absurd, treat as invalid detection
        if theta_deg is not None and abs(float(theta_deg)) > self.cfg.theta_hard_limit_deg:
            valid_lane = False

        # SOC scaling
        scale = 1.0
        if self.cfg.use_soc_scaling and (soc is not None):
            s = _clip(float(soc), 0.0, 1.0)
            scale = _clip(self.cfg.soc_floor + (1.0 - self.cfg.soc_floor) * s,
                          self.cfg.soc_floor, 1.0)

        mode = (lane_mode or "normal").lower()

        # ---------------- normal ----------------
        if mode == "normal":
            if not (valid_lane and (lateral_px is not None) and (theta_deg is not None)):
                mode = "hold"
            else:
                lateral_n = float(lateral_px) / self._half_w
                theta_rad = math.radians(float(theta_deg))

                # Calculate steer angle directly
                steer = self.cfg.k_theta * theta_rad + self.cfg.k_lateral * lateral_n
                steer = _clip(steer, -self.cfg.steer_limit, self.cfg.steer_limit)

                # Slow down in curves based on steer angle
                steer_cost = abs(steer) / self.cfg.steer_limit
                base = self.cfg.v_max * max(0.0, 1.0 - steer_cost * 0.7)
                base = max(self.cfg.v_min, min(self.cfg.v_max, base))

                # Ensure a minimum base speed when cornering hard
                corner_base_floor = 0.28
                theta_thr_deg = 15.0
                if abs(theta_deg) >= theta_thr_deg:
                    base = max(base, corner_base_floor)

                self._last_base = base
                self._last_steer = steer

                drive = base * scale * self.cfg.forward_sign
                drive, steer = self._post_process(drive, steer)
                self._store_debug(True, True, "normal", lateral_px, theta_deg,
                                  lateral_n, theta_rad, steer, drive, steer, soc, base, scale, lost_age)
                return drive, steer

        # ---------------- hold ----------------
        if mode == "hold":
            decay = self.cfg.hold_decay_per_frame ** max(0, int(lost_age))
            base = self._last_base * decay
            steer = self._last_steer * decay
            drive = base * scale * self.cfg.forward_sign
            drive, steer = self._post_process(drive, steer)
            self._store_debug(False, True, "hold", lateral_px, theta_deg,
                              0.0, 0.0, steer, drive, steer, soc, base, scale, lost_age)
            return drive, steer

        # ---------------- search ----------------
        base = _clip(self.cfg.search_speed, self.cfg.v_min, self.cfg.v_max) if not self.cfg.search_pivot else 0.0
        steer = abs(self.cfg.search_steer_const)  # constant right turn
        drive = base * scale * self.cfg.forward_sign
        drive, steer = self._post_process(drive, steer)
        self._store_debug(False, True, "search", lateral_px, theta_deg,
                          0.0, 0.0, steer, drive, steer, soc, base, scale, lost_age)
        return drive, steer

    # ------------ helpers ------------
    def _post_process(self, drive: float, steer: float) -> Tuple[float, float]:
        """Apply limits and smooth."""
        drive = _clip(drive, -self.cfg.torque_limit, self.cfg.torque_limit)
        steer = _clip(steer, -self.cfg.steer_limit, self.cfg.steer_limit)
        return self._smooth(drive, steer)

    def _smooth(self, drive: float, steer: float) -> Tuple[float, float]:
        """IIR smoothing on output commands."""
        a = _clip(self.cfg.alpha_smooth, 0.0, 1.0)
        drive_s = (1 - a) * drive + a * self._prev_drive
        steer_s = (1 - a) * steer + a * self._prev_steer
        self._prev_drive, self._prev_steer = drive_s, steer_s
        return drive_s, steer_s

    def _store_debug(
        self,
        valid_lane: bool,
        start_go: bool,
        lane_mode: str,
        lateral_px: Optional[float],
        theta_deg: Optional[float],
        lateral_n: float,
        theta_rad: float,
        steer_cmd: float,
        drive: float,
        steer: float,
        soc: Optional[float],
        base: Optional[float],
        scale: Optional[float],
        lost_age: int,
    ) -> None:
        """Save last-frame debug values (read by overlay)."""
        self.last_debug = {
            "start_go": bool(start_go),
            "valid_lane": bool(valid_lane),
            "lane_mode": str(lane_mode),
            "lost_age": int(lost_age),
            "lateral_px": None if lateral_px is None else float(lateral_px),
            "theta_deg": None if theta_deg is None else float(theta_deg),
            "lateral_norm": float(lateral_n),
            "theta_rad": float(theta_rad),
            "steer_cmd": float(steer_cmd),
            "base_speed": None if base is None else float(base),
            "soc": None if soc is None else float(soc),
            "soc_scale": None if scale is None else float(scale),
            "drive_torque": float(drive),
            "steer_angle": float(steer),
            "half_width_px": float(self._half_w),
            "forward_sign": int(self.cfg.forward_sign),
            "loop_period_s": float(self.cfg.loop_period_s),
        }


# ------------- optional CLI test -------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_width", type=int, default=224)
    ap.add_argument("--lateral_px", type=float, default=3.0)
    ap.add_argument("--theta_deg", type=float, default=2.0)
    ap.add_argument("--soc", type=float, default=1.0)
    ap.add_argument("--start_go", action="store_true")
    ap.add_argument("--valid_lane", action="store_true")
    ap.add_argument("--lane_mode", type=str, default="normal")
    ap.add_argument("--lost_age", type=int, default=0)
    ap.add_argument("--forward_sign", type=int, default=+1, choices=[-1, +1])
    args = ap.parse_args()

    cfg = DriverConfig(image_width=args.image_width, forward_sign=args.forward_sign)
    drv = DriverModel(cfg)
    drive, steer = drv.update(
        lateral_px=args.lateral_px,
        theta_deg=args.theta_deg,
        soc=args.soc,
        image_width=args.image_width,
        start_go=args.start_go,
        valid_lane=args.valid_lane,
        lane_mode=args.lane_mode,
        lost_age=args.lost_age,
    )
    print(f"drive={drive:+.3f}, steer={steer:+.3f} rad ({math.degrees(steer):+.1f} deg)")
    print("debug:", drv.last_debug)
