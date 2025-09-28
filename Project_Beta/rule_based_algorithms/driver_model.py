# rule_based_algorithms/driver_model.py
# ----------------------------------------------------------------------
# Higher layer decides lane_mode ("normal" / "hold" / "search") and lost_age.
# This driver ONLY computes torques. Start(GO) gating and safety are handled here.
#
# Sign conventions:
#   lateral_px : right is + [px] (offset from image center)
#   theta_deg  : 0° is up, tilting to the right is + [deg]
#   Output     : + means forward when forward_sign=+1
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
    k_theta: float = 0.45
    k_lateral: float = 0.40
    yaw_mix_gain: float = 0.50

    # Output shaping
    torque_limit: float = 1.00
    alpha_smooth: float = 0.65  # 0: off, 1: heavy smoothing (IIR)

    # Anti-pivot guard (keep outer wheel driving forward; allow limited negative inner)
    anti_pivot_limit: float = 0.07   # min inner (allows down to -0.30 if you raise this)
    min_outer_torque: float = 0.10   # floor for the outer wheel

    # Battery (SOC) scaling
    use_soc_scaling: bool = True
    soc_floor: float = 0.30

    # Conventions / safety
    forward_sign: int = +1
    theta_hard_limit_deg: float = 80.0
    invalid_brake: float = 0.0        # output when NO-GO etc. (0.0 recommended)
    require_start_go: bool = True     # True: must stop if NO-GO

    allow_pivot: bool = True          # True to disable non-negative clamp in _mix_lr (not used here)
    yaw_clip_margin: float = 0.01     # small margin to avoid sticking exactly at 0

    # ====== Lane-lost related (pure math parameters; mode decision is upstream) ======
    hold_decay_per_frame: float = 0.90   # decay ratio during hold
    search_pivot: bool = True            # True: pivot in place / False: slow forward while turning
    search_speed: float = 0.00           # forward component (0.0 when pivoting)
    search_yaw_const: float = 0.18       # [rad] constant yaw to clockwise (right)
    loop_period_s: float = 0.050         # for logs only


class DriverModel:
    """Given lane_mode, produce left/right torques only (no I/O here)."""

    def __init__(self, cfg: DriverConfig):
        self.cfg = cfg
        self._prev_left = 0.0
        self._prev_right = 0.0
        self._half_w = max(
            1.0,
            (cfg.lateral_norm_halfwidth_px
             if cfg.lateral_norm_halfwidth_px is not None
             else cfg.image_width * 0.5)
        )
        # For hold mode, remember most recent base/yaw
        self._last_base = 0.0
        self._last_yaw  = 0.0
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
        """Compute (left, right) torques for the current frame."""
        # Geometry update
        if image_width is not None and image_width > 0:
            self._half_w = max(1.0, image_width * 0.5)

        # Start gate
        if self.cfg.require_start_go and not start_go:
            lt = rt = self.cfg.invalid_brake * self.cfg.forward_sign
            lt, rt = self._smooth(lt, rt)
            self._store_debug(False, False, lane_mode, lateral_px, theta_deg,
                              0.0, 0.0, 0.0, lt, rt, soc, None, None, lost_age)
            return lt, rt

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
                yaw = self.cfg.k_theta * theta_rad + self.cfg.k_lateral * lateral_n

                # Slow down in curves
                steer_cost = (self.cfg.slow_w_theta * abs(theta_rad) +
                              self.cfg.slow_w_lateral * abs(lateral_n))
                base = self.cfg.v_max * max(0.0, 1.0 - steer_cost)
                base = max(self.cfg.v_min, min(self.cfg.v_max, base))

                # Ensure a minimum base speed when cornering hard
                corner_base_floor = 0.28
                theta_thr_deg = 15.0
                if abs(theta_deg) >= theta_thr_deg:
                    base = max(base, corner_base_floor)

                self._last_base = base
                self._last_yaw  = yaw

                left, right = self._mix_lr(base, yaw, scale)
                left, right = self._post_process(left, right, yaw)
                self._store_debug(True, True, "normal", lateral_px, theta_deg,
                                  lateral_n, theta_rad, yaw, left, right, soc, base, scale, lost_age)
                return left, right

        # ---------------- hold ----------------
        if mode == "hold":
            decay = self.cfg.hold_decay_per_frame ** max(0, int(lost_age))
            base = self._last_base * decay
            yaw  = self._last_yaw  * decay
            left, right = self._mix_lr(base, yaw, scale)
            left, right = self._post_process(left, right, yaw)
            self._store_debug(False, True, "hold", lateral_px, theta_deg,
                              0.0, 0.0, yaw, left, right, soc, base, scale, lost_age)
            return left, right

        # ---------------- search ----------------
        base = _clip(self.cfg.search_speed, self.cfg.v_min, self.cfg.v_max) if not self.cfg.search_pivot else 0.0
        yaw  = abs(self.cfg.search_yaw_const)  # clockwise
        left, right = self._mix_lr(base, yaw, scale)
        left, right = self._post_process(left, right, yaw)
        self._store_debug(False, True, "search", lateral_px, theta_deg,
                          0.0, 0.0, yaw, left, right, soc, base, scale, lost_age)
        return left, right

    # ------------ helpers ------------
    def _mix_lr(self, base: float, yaw: float, scale: float) -> Tuple[float, float]:
        """Blend base & yaw into left/right torques (before limits/smoothing)."""
        fs = self.cfg.forward_sign
        y = self.cfg.yaw_mix_gain * yaw

        # Reduce yaw at low speed (empirical)
        # current: 0.3 + 0.7 * (base / v_max)
        yaw_scale = 0.3 + 0.7 * (base / max(1e-6, self.cfg.v_max))
        y *= yaw_scale

        # No non-neg clamp here (inner can go slightly negative by design)
        left  = (base + y) * scale * fs
        right = (base - y) * scale * fs
        return left, right

    def _post_process(self, left: float, right: float, yaw: float) -> Tuple[float, float]:
        """Apply torque limits and anti-pivot guard, then smooth."""
        left  = _clip(left,  -self.cfg.torque_limit, self.cfg.torque_limit)
        right = _clip(right, -self.cfg.torque_limit, self.cfg.torque_limit)

        apl = getattr(self.cfg, "anti_pivot_limit", None)  # e.g., 0.07–0.15
        mot = getattr(self.cfg, "min_outer_torque", 0.0)   # e.g., 0.05–0.10

        if apl is not None:
            if yaw >= 0.0:          # turning right
                left  = max(left,  +mot)   # outer must drive forward
                right = max(right, -apl)   # inner allowed down to -apl
            else:                   # turning left
                right = max(right, +mot)
                left  = max(left,  -apl)

        return self._smooth(left, right)

    def _smooth(self, lt: float, rt: float) -> Tuple[float, float]:
        """IIR smoothing on output torques."""
        a = _clip(self.cfg.alpha_smooth, 0.0, 1.0)
        lt_s = (1 - a) * lt + a * self._prev_left
        rt_s = (1 - a) * rt + a * self._prev_right
        self._prev_left, self._prev_right = lt_s, rt_s
        return lt_s, rt_s

    def _store_debug(
        self,
        valid_lane: bool,
        start_go: bool,
        lane_mode: str,
        lateral_px: Optional[float],
        theta_deg: Optional[float],
        lateral_n: float,
        theta_rad: float,
        yaw: float,
        left: float,
        right: float,
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
            "yaw_cmd": float(yaw),
            "base_speed": None if base is None else float(base),
            "soc": None if soc is None else float(soc),
            "soc_scale": None if scale is None else float(scale),
            "left_torque": float(left),
            "right_torque": float(right),
            "half_width_px": float(self._half_w),
            "forward_sign": int(self.cfg.forward_sign),
            "loop_period_s": float(self.cfg.loop_period_s),
            # extra for downstream debug/overlays
            "apl": float(getattr(self.cfg, "anti_pivot_limit", -1.0)),
            "mot": float(getattr(self.cfg, "min_outer_torque", -1.0)),
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
    lt, rt = drv.update(
        lateral_px=args.lateral_px,
        theta_deg=args.theta_deg,
        soc=args.soc,
        image_width=args.image_width,
        start_go=args.start_go,
        valid_lane=args.valid_lane,
        lane_mode=args.lane_mode,
        lost_age=args.lost_age,
    )
    print(f"left={lt:+.3f}, right={rt:+.3f}")
    print("debug:", drv.last_debug)
