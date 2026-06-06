#!/usr/bin/env python3
# table_model.py
# ============================================================
# Phase 1: System identification for the table-mode model-based pipeline.
#
# Reads all Robot1/training_data/run_*/metadata.csv and identifies a
# kinematic bicycle model from the recorded trajectories + inputs:
#
#   (A) Steering -> yaw:   yaw_rate = (v / L) * tan(steer)   -> identify L
#   (B) Speed dynamics:    v_dot    = a*torque - b*v - c     -> identify a,b,c
#   (C) Latency:           cross-correlation lag between steer command
#                          and yaw-rate response (in 50 ms frames)
#
# Outputs identified params + fit quality (R^2) and writes a per-run
# trajectory summary. No Unity required.
#
# Usage:  python scripts/table_model.py
# ============================================================

import sys
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "Robot1" / "training_data"
OUT_JSON = Path(__file__).resolve().parent / "table_model_params.json"

DT = 0.05  # 50 ms control period

# statuses that count as "moving on track"
RUN_STATUSES = {"Lap0", "Lap1", "Lap2", "Lap3", "Running"}


def wrap_deg(d):
    """Wrap degrees to (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def load_run(meta_path: Path):
    """Load a run's metadata, return a clean moving-section DataFrame or None."""
    try:
        df = pd.read_csv(meta_path)
    except Exception:
        return None
    if df.empty or "status" not in df.columns:
        return None
    # keep only the moving section
    df = df[df["race_time_ms"] > 0].copy()
    df = df[df["status"].isin(RUN_STATUSES)].copy()
    if len(df) < 30:
        return None
    df = df.reset_index(drop=True)
    return df


def derive_kinematics(df: pd.DataFrame):
    """Compute v, yaw_rate (rad/s), and longitudinal accel from a run."""
    z = df["pos_z"].to_numpy(dtype=float)
    x = df["pos_x"].to_numpy(dtype=float)
    yaw = df["yaw"].to_numpy(dtype=float)
    torque = df["drive_torque"].to_numpy(dtype=float)
    steer = df["steer_angle"].to_numpy(dtype=float)  # radians

    dz = np.diff(z)
    dx = np.diff(x)
    speed = np.sqrt(dz * dz + dx * dx) / DT  # m/s, length N-1

    dyaw = wrap_deg(np.diff(yaw))
    yaw_rate = np.radians(dyaw) / DT  # rad/s, length N-1

    # midpoint-aligned inputs for the N-1 intervals
    steer_mid = steer[:-1]
    torque_mid = torque[:-1]
    speed_mid = speed  # already interval-based

    # longitudinal accel over interval (length N-2)
    v_dot = np.diff(speed) / DT

    return {
        "speed": speed,
        "yaw_rate": yaw_rate,
        "steer": steer_mid,
        "torque": torque_mid,
        "v_dot": v_dot,
    }


def r2(y, y_hat):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main():
    if not TRAIN_DIR.exists():
        print(f"[Model] ERROR: {TRAIN_DIR} not found.")
        sys.exit(1)

    runs = sorted(d for d in TRAIN_DIR.iterdir() if d.is_dir() and d.name.startswith("run_"))
    print(f"[Model] Found {len(runs)} run folders.")

    # accumulators for global regression
    A_steer_x, A_steer_y = [], []        # for (A): x = v*tan(steer), y = yaw_rate
    B_feat, B_target = [], []            # for (B): features [torque, v, 1], target v_dot
    steer_series, yawrate_series = [], []  # for (C): latency cross-correlation

    summary = []
    used = 0
    for run in runs:
        meta = run / "metadata.csv"
        if not meta.exists():
            continue
        df = load_run(meta)
        if df is None:
            summary.append({"run": run.name, "ok": False})
            continue
        used += 1
        k = derive_kinematics(df)

        # speed-validity mask (avoid near-zero speed where tan/ratios are noisy)
        v = k["speed"]
        valid = v > 0.05

        # (A) steering -> yaw
        x_steer = v[valid] * np.tan(k["steer"][valid])
        y_yaw = k["yaw_rate"][valid]
        A_steer_x.append(x_steer)
        A_steer_y.append(y_yaw)

        # (B) speed dynamics (length N-2): align inputs to v_dot intervals
        vdot = k["v_dot"]
        tq = k["torque"][:-1]
        vv = v[:-1]
        m2 = vv > 0.0
        B_feat.append(np.column_stack([tq[m2], vv[m2], np.ones(np.sum(m2))]))
        B_target.append(vdot[m2])

        # (C) latency series (whole run, raw)
        steer_series.append(k["steer"])
        yawrate_series.append(k["yaw_rate"])

        # per-run summary
        race_s = df["race_time_ms"].max() / 1000.0
        finished = bool(df["status"].isin(["Finish", "Finished"]).any())
        summary.append({
            "run": run.name,
            "ok": True,
            "rows": int(len(df)),
            "race_time_s": round(race_s, 1),
            "v_mean": round(float(np.mean(v)), 3),
            "v_max": round(float(np.max(v)), 3),
            "torque_mean": round(float(np.mean(k["torque"])), 3),
            "steer_abs_mean": round(float(np.mean(np.abs(k["steer"]))), 4),
        })

    if used == 0:
        print("[Model] ERROR: no usable runs.")
        sys.exit(1)

    # ---- (A) identify L : yaw_rate = (1/L) * (v*tan(steer)) ----
    ax = np.concatenate(A_steer_x)
    ay = np.concatenate(A_steer_y)
    # least squares through origin: ay = slope * ax
    slope = float(np.dot(ax, ay) / np.dot(ax, ax))
    L = 1.0 / slope if slope != 0 else float("nan")
    a_r2 = r2(ay, slope * ax)

    # ---- (B) identify a,b,c : v_dot = a*torque - b*v - c ----
    Bf = np.vstack(B_feat)
    Bt = np.concatenate(B_target)
    coef, *_ = np.linalg.lstsq(Bf, Bt, rcond=None)
    a_drive, neg_b, neg_c = coef
    b_drag = -neg_b
    c_roll = -neg_c
    b_r2 = r2(Bt, Bf @ coef)

    # ---- (C) latency cross-correlation (steer leads yaw_rate) ----
    s_all = np.concatenate(steer_series)
    y_all = np.concatenate(yawrate_series)
    s_all = s_all - np.mean(s_all)
    y_all = y_all - np.mean(y_all)
    best_lag, best_corr = 0, -1e9
    for lag in range(0, 8):  # 0..7 frames (0..350 ms)
        if lag == 0:
            c = float(np.dot(s_all, y_all))
        else:
            c = float(np.dot(s_all[:-lag], y_all[lag:]))
        denom = (np.linalg.norm(s_all) * np.linalg.norm(y_all)) + 1e-9
        c_norm = c / denom
        if c_norm > best_corr:
            best_corr, best_lag = c_norm, lag

    params = {
        "dt": DT,
        "runs_used": used,
        "bicycle": {
            "wheelbase_L": round(L, 4),
            "steer_to_yaw_R2": round(a_r2, 3),
            "_model": "yaw_rate = (v / L) * tan(steer)",
        },
        "speed": {
            "a_drive": round(float(a_drive), 4),
            "b_drag": round(float(b_drag), 4),
            "c_roll": round(float(c_roll), 4),
            "speed_R2": round(b_r2, 3),
            "_model": "v_dot = a_drive*torque - b_drag*v - c_roll",
            "v_terminal_full_throttle": round(float((a_drive - c_roll) / b_drag), 3) if b_drag > 0 else None,
        },
        "latency": {
            "best_lag_frames": best_lag,
            "best_lag_ms": best_lag * int(DT * 1000),
            "norm_xcorr": round(best_corr, 3),
        },
    }

    print("\n========== SYSTEM IDENTIFICATION RESULTS ==========")
    print(json.dumps(params, indent=2))

    print("\n========== PER-RUN SUMMARY ==========")
    print(f"{'run':<26}{'time_s':>8}{'v_mean':>8}{'v_max':>8}{'tq_mean':>9}{'|steer|':>9}")
    for s in summary:
        if not s.get("ok"):
            print(f"{s['run']:<26}{'  (skipped)':>8}")
            continue
        print(f"{s['run']:<26}{s['race_time_s']:>8}{s['v_mean']:>8}{s['v_max']:>8}"
              f"{s['torque_mean']:>9}{s['steer_abs_mean']:>9}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"params": params, "summary": summary}, f, indent=2)
    print(f"\n[Model] Saved -> {OUT_JSON.name}")


if __name__ == "__main__":
    main()
