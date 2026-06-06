#!/usr/bin/env python3
# make_table_csv.py
# ============================================================
# Phase 2 (offline part): generate Robot1/table_input.csv from a
# reference run using the identified kinematic model (differential
# flatness inverse for steering).
#
# Modes:
#   replay    : recorded drive_torque + recorded steer_angle
#               (pure open-loop replay -> reproducibility baseline)
#   flatsteer : recorded drive_torque + MODEL steer (delta = atan(L*kappa))
#               recomputed from the reference line via differential flatness
#   model     : speed-profile torque (corner speed cap + accel/brake limits)
#               + model steer
#
# Before writing, the script runs a SELF-CHECK: it reconstructs the
# reference run's own steering from its own path and compares to the
# recorded steer_angle (RMSE / correlation). A good match validates the
# inverse-model approach offline, before any Unity run.
#
# Usage:
#   python scripts/make_table_csv.py --mode flatsteer
#   python scripts/make_table_csv.py --run run_20260506_165600 --mode model
# ============================================================

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "Robot1" / "training_data"
LIVE_CSV = BASE_DIR / "Robot1" / "table_input.csv"
GEN_DIR = Path(__file__).resolve().parent / "generated_tables"
PARAMS_JSON = Path(__file__).resolve().parent / "table_model_params.json"

DT = 0.05
STEER_LIMIT = 0.524  # rad (~30 deg, matches Unity)


def load_params():
    if PARAMS_JSON.exists():
        with open(PARAMS_JSON, encoding="utf-8") as f:
            return json.load(f)["params"]
    # fallback defaults if sysID not run
    return {
        "bicycle": {"wheelbase_L": 0.274},
        "speed": {"a_drive": 13.2, "b_drag": 10.96, "c_roll": -0.58},
    }


def pick_run(run_name: str | None) -> Path:
    if run_name:
        p = TRAIN_DIR / run_name
        if not p.exists():
            raise SystemExit(f"[Gen] run not found: {p}")
        return p
    runs = sorted(d for d in TRAIN_DIR.iterdir() if d.is_dir() and d.name.startswith("run_"))
    if not runs:
        raise SystemExit("[Gen] no runs found.")
    return runs[-1]


def moving_avg(a, w=5):
    if w <= 1:
        return a
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="same")


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def compute_model_steer(df: pd.DataFrame, L: float, smooth_w: int = 5):
    """Differential-flatness steering from the reference line.

    kappa magnitude from path geometry; sign from recorded yaw-rate.
    delta = atan(L * kappa_signed)
    Returns array aligned to df rows (len N), 0 where speed is ~0.
    """
    z = moving_avg(df["pos_z"].to_numpy(float), smooth_w)
    x = moving_avg(df["pos_x"].to_numpy(float), smooth_w)
    yaw = df["yaw"].to_numpy(float)
    n = len(z)

    # first/second derivatives (central differences) w.r.t. time
    zp = np.gradient(z, DT)
    xp = np.gradient(x, DT)
    zpp = np.gradient(zp, DT)
    xpp = np.gradient(xp, DT)

    speed = np.sqrt(zp * zp + xp * xp)
    denom = np.power(zp * zp + xp * xp, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa_mag = np.abs(zp * xpp - xp * zpp) / denom
    kappa_mag = np.nan_to_num(kappa_mag, nan=0.0, posinf=0.0, neginf=0.0)

    # sign from recorded yaw-rate (trusted: steer->yaw fit R^2=0.81)
    dyaw = np.zeros(n)
    dyaw[1:] = wrap_deg(np.diff(yaw))
    yaw_rate = np.radians(dyaw) / DT
    sign = np.sign(moving_avg(yaw_rate, smooth_w))

    kappa_signed = sign * kappa_mag
    delta = np.arctan(L * kappa_signed)

    # zero out near-stationary frames (curvature undefined)
    delta[speed < 0.05] = 0.0
    delta = np.clip(delta, -STEER_LIMIT, STEER_LIMIT)
    return delta, speed, kappa_mag, yaw_rate


def speed_profile_torque(df, params, kappa_mag, a_lat_max):
    """Generate a torque column from a corner-speed-capped velocity profile.

    v_cap(corner) = sqrt(a_lat_max / kappa)  (friction-circle lite)
    forward/backward passes apply accel/brake limits, then invert the
    identified speed model to get torque.
    """
    a = params["speed"]["a_drive"]
    b = params["speed"]["b_drag"]
    c = params["speed"]["c_roll"]
    n = len(df)

    v_term = (a - c) / b if b > 0 else 1.2  # full-throttle terminal speed
    with np.errstate(divide="ignore"):
        v_cap = np.where(kappa_mag > 1e-4, np.sqrt(a_lat_max / np.maximum(kappa_mag, 1e-4)), v_term)
    v_cap = np.minimum(v_cap, v_term)

    # accel/brake limits from model extremes
    a_accel = a * 1.0 - c        # approx max accel near v=0 at full throttle
    a_brake = a * 1.0 + c        # approx max decel at full reverse (rough)
    a_accel = max(a_accel, 1.0)
    a_brake = max(a_brake, 1.0)

    v = v_cap.copy()
    # forward pass (accel limit)
    for i in range(1, n):
        v[i] = min(v[i], v[i - 1] + a_accel * DT)
    # backward pass (brake limit)
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], v[i + 1] + a_brake * DT)

    v_dot = np.gradient(v, DT)
    torque = (v_dot + b * v + c) / a
    torque = np.clip(torque, -1.0, 1.0)
    # keep stationary during countdown
    torque[df["drive_torque"].to_numpy(float) == 0.0] = 0.0
    return torque, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_20260506_165600")
    ap.add_argument("--mode", choices=["replay", "flatsteer", "model"], default="flatsteer")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--write-live", action="store_true",
                    help="overwrite Robot1/table_input.csv (the file aira reads)")
    args = ap.parse_args()

    params = load_params()
    L = params["bicycle"]["wheelbase_L"]

    run_dir = pick_run(args.run)
    meta = run_dir / "metadata.csv"
    df = pd.read_csv(meta)
    # full sequence up to (and including) Finish to preserve start timing
    fin = df.index[df["status"].isin(["Finish", "Finished"])]
    end = int(fin[0]) + 1 if len(fin) else len(df)
    df = df.iloc[:end].reset_index(drop=True)
    n = len(df)

    rec_torque = df["drive_torque"].to_numpy(float)
    rec_steer = df["steer_angle"].to_numpy(float)

    # --- model steer + self-check (always) ---
    delta, speed, kappa_mag, yaw_rate = compute_model_steer(df, L, args.smooth)
    moving = speed > 0.1
    if moving.sum() > 10:
        rmse = float(np.sqrt(np.mean((delta[moving] - rec_steer[moving]) ** 2)))
        corr = float(np.corrcoef(delta[moving], rec_steer[moving])[0, 1])
    else:
        rmse, corr = float("nan"), float("nan")
    a_lat = np.abs(speed * yaw_rate)  # |v * yaw_rate| = lateral accel
    a_lat_max = float(np.percentile(a_lat[moving], 90)) if moving.sum() else 1.0

    print(f"[Gen] Reference run : {run_dir.name}  (rows={n}, L={L} m)")
    print(f"[Gen] SELF-CHECK (model steer vs recorded steer, moving section):")
    print(f"        RMSE = {rmse:.4f} rad ({math.degrees(rmse):.2f} deg)")
    print(f"        corr = {corr:.3f}")
    print(f"        recorded |steer| mean = {np.mean(np.abs(rec_steer[moving])):.4f} rad")
    print(f"[Gen] lateral accel (90th pct) a_lat_max = {a_lat_max:.3f} m/s^2")

    # --- assemble output columns by mode ---
    if args.mode == "replay":
        out_torque, out_steer = rec_torque, rec_steer
    elif args.mode == "flatsteer":
        out_torque, out_steer = rec_torque, delta
    else:  # model
        out_torque, out_speed = speed_profile_torque(df, params, kappa_mag, a_lat_max)
        out_steer = delta
        print(f"[Gen] model speed profile: v_target mean={np.mean(out_speed):.3f} "
              f"max={np.max(out_speed):.3f} m/s; torque mean={np.mean(np.abs(out_torque)):.3f}")

    out = pd.DataFrame({
        "time_id": np.arange(n),
        "drive_torque": np.round(out_torque, 3),
        "steer_angle": np.round(out_steer, 3),
    })

    GEN_DIR.mkdir(exist_ok=True)
    gen_path = GEN_DIR / f"{run_dir.name}_{args.mode}.csv"
    out.to_csv(gen_path, index=False)
    print(f"[Gen] Wrote {len(out)} rows -> {gen_path.relative_to(BASE_DIR)}")

    if args.write_live:
        out.to_csv(LIVE_CSV, index=False)
        print(f"[Gen] Wrote LIVE -> {LIVE_CSV.relative_to(BASE_DIR)}  (aira will read this)")
    else:
        print(f"[Gen] (dry) to use it in aira: rerun with --write-live")


if __name__ == "__main__":
    main()
