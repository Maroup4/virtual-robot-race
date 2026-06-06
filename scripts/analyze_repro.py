#!/usr/bin/env python3
# analyze_repro.py
# ============================================================
# Reproducibility analysis for open-loop table-mode replays.
#
# Loads a set of runs (default: today's run_20260606_*) plus the
# reference run, and quantifies how much the trajectories diverge
# despite using the IDENTICAL table_input.csv (open-loop nondeterminism).
#
# Metrics:
#   - per-run: final status, finished?, race_time, collisions, rows
#   - cross-run trajectory spread [m]: interpolate each run's (pos_z,pos_x)
#     onto a common race_time grid, then std across runs at each time
#   - deviation from the reference line [m]
#
# Usage:  python scripts/analyze_repro.py [--prefix run_20260606]
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "Robot1" / "training_data"
REF_RUN = "run_20260506_165600"

RUN_STATUSES = {"Lap0", "Lap1", "Lap2", "Lap3", "Running"}


def load_moving(run_dir: Path):
    meta = run_dir / "metadata.csv"
    if not meta.exists():
        return None
    df = pd.read_csv(meta)
    if df.empty or "status" not in df.columns:
        return None
    mv = df[df["race_time_ms"] > 0].copy()
    mv = mv[mv["status"].isin(RUN_STATUSES | {"Finish", "Finished"})].copy()
    return df, mv.reset_index(drop=True)


def summarize(name, df, mv):
    finished = bool(df["status"].isin(["Finish", "Finished"]).any())
    last_status = str(df["status"].iloc[-1])
    race_s = float(df["race_time_ms"].max()) / 1000.0
    if "collision_type" in df.columns:
        col = int((df["collision_type"].fillna("") != "").sum())
    else:
        col = 0
    final_soc = float(df["soc"].iloc[-1]) if "soc" in df.columns else float("nan")
    return {
        "run": name,
        "finished": finished,
        "last_status": last_status,
        "race_s": round(race_s, 1),
        "collisions": col,
        "final_soc": round(final_soc, 2),
        "moving_rows": int(len(mv)),
    }


def resample_path(mv, t_grid):
    """Interpolate (pos_z, pos_x) onto common race-time grid (ms)."""
    t = mv["race_time_ms"].to_numpy(float)
    z = mv["pos_z"].to_numpy(float)
    x = mv["pos_x"].to_numpy(float)
    # ensure strictly increasing t
    order = np.argsort(t)
    t, z, x = t[order], z[order], x[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, z, x = t[keep], z[keep], x[keep]
    zi = np.interp(t_grid, t, z, left=np.nan, right=np.nan)
    xi = np.interp(t_grid, t, x, left=np.nan, right=np.nan)
    return zi, xi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="run_20260606")
    args = ap.parse_args()

    run_dirs = sorted(d for d in TRAIN_DIR.iterdir()
                      if d.is_dir() and d.name.startswith(args.prefix))
    if not run_dirs:
        raise SystemExit(f"[Repro] no runs matching {args.prefix}")

    print(f"[Repro] Replay runs: {len(run_dirs)} (prefix {args.prefix})")
    print(f"[Repro] Reference  : {REF_RUN}\n")

    loaded = []
    summaries = []
    for d in run_dirs:
        r = load_moving(d)
        if r is None:
            continue
        df, mv = r
        loaded.append((d.name, mv))
        summaries.append(summarize(d.name, df, mv))

    # reference
    ref = load_moving(TRAIN_DIR / REF_RUN)
    ref_mv = ref[1] if ref else None
    if ref:
        summaries.append({**summarize(REF_RUN + " (ref)", ref[0], ref[1])})

    # --- per-run summary table ---
    print(f"{'run':<34}{'fin':>5}{'status':>10}{'time_s':>8}{'col':>5}{'soc':>6}{'rows':>6}")
    for s in summaries:
        print(f"{s['run']:<34}{str(s['finished']):>5}{s['last_status']:>10}"
              f"{s['race_s']:>8}{s['collisions']:>5}{s['final_soc']:>6}{s['moving_rows']:>6}")

    # --- cross-run trajectory spread on common time grid ---
    # common window = up to the shortest run's end time
    end_times = [mv["race_time_ms"].max() for _, mv in loaded]
    if not end_times:
        return
    t_max = float(min(end_times))
    t_grid = np.arange(0.0, t_max, 50.0)  # 50 ms grid

    Zs, Xs = [], []
    for _, mv in loaded:
        zi, xi = resample_path(mv, t_grid)
        Zs.append(zi)
        Xs.append(xi)
    Z = np.vstack(Zs)
    X = np.vstack(Xs)

    valid = ~np.isnan(Z).any(axis=0) & ~np.isnan(X).any(axis=0)
    Zv, Xv = Z[:, valid], X[:, valid]

    # per-timestep euclidean spread across runs (std of point cloud)
    z_mean = Zv.mean(axis=0)
    x_mean = Xv.mean(axis=0)
    # RMS distance of each run's point from the mean point, per timestep
    dist = np.sqrt((Zv - z_mean) ** 2 + (Xv - x_mean) ** 2)
    spread_per_t = dist.mean(axis=0)  # mean over runs at each time

    print("\n[Repro] Open-loop trajectory spread across identical-CSV replays:")
    print(f"        common window      : 0 - {t_max/1000:.1f} s ({valid.sum()} pts)")
    print(f"        mean spread        : {spread_per_t.mean():.4f} m")
    print(f"        max  spread        : {spread_per_t.max():.4f} m")
    print(f"        spread @ end window : {spread_per_t[-1]:.4f} m")

    # --- deviation from reference line ---
    if ref_mv is not None:
        rz, rx = resample_path(ref_mv, t_grid)
        rv = (~np.isnan(rz)) & (~np.isnan(rx)) & valid
        if rv.sum() > 5:
            dev = np.sqrt((Zv[:, rv[valid]] - rz[rv]) ** 2 + (Xv[:, rv[valid]] - rx[rv]) ** 2)
            print(f"\n[Repro] Deviation of replays from REFERENCE line:")
            print(f"        mean dev : {dev.mean():.4f} m")
            print(f"        max  dev : {dev.max():.4f} m")

    # time spread
    times = [s["race_s"] for s in summaries if "(ref)" not in s["run"]]
    print(f"\n[Repro] Race-time spread: min={min(times):.1f}s max={max(times):.1f}s "
          f"mean={np.mean(times):.1f}s std={np.std(times):.2f}s")
    fin_n = sum(1 for s in summaries if "(ref)" not in s["run"] and s["finished"])
    print(f"[Repro] Finished: {fin_n}/{len(times)}")


if __name__ == "__main__":
    main()
