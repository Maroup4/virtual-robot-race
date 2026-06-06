#!/usr/bin/env python3
# optimize_params.py
# Hill-climbing parameter optimizer for rule-based control.
# Called between episodes by headless_loop.py:
#   python scripts/headless_loop.py 30 --retrain "python scripts/optimize_params.py"
#
# Flow per call:
#   1. Find the latest run's metadata.csv
#   2. Compute score (lower = better): race_time + collision penalty
#   3. Compare to best known score; keep or revert
#   4. Perturb one parameter
#   5. Write updated params into Robot1/rule_based_input.py

import json
import re
import random
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("[Optimizer] ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = Path(__file__).resolve().parent / "optimization_state.json"
RULE_BASED_INPUT = BASE_DIR / "Robot1" / "rule_based_input.py"

# Parameter search space: name -> (min, max, step)
PARAM_SPACE = {
    "v_max":        (0.50, 1.00, 0.05),
    "k_theta":      (0.30, 1.50, 0.10),
    "k_lateral":    (0.20, 1.00, 0.10),
    "alpha_smooth": (0.10, 0.60, 0.05),
}

DEFAULT_PARAMS = {
    "v_max":        0.75,
    "k_theta":      0.90,
    "k_lateral":    0.60,
    "alpha_smooth": 0.30,
}

DNF_SCORE = 999999.0


def find_latest_run() -> Path | None:
    training_data = BASE_DIR / "Robot1" / "training_data"
    if not training_data.exists():
        return None
    runs = sorted(
        [d for d in training_data.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )
    return runs[-1] if runs else None


def compute_score(run_dir: Path) -> tuple[float, str]:
    """Compute performance score. Lower is better. Returns (score, reason)."""
    meta = run_dir / "metadata.csv"
    if not meta.exists():
        return DNF_SCORE, "no_metadata"

    try:
        df = pd.read_csv(meta)
    except Exception as e:
        return DNF_SCORE, f"csv_error:{e}"

    if df.empty or "status" not in df.columns:
        return DNF_SCORE, "empty_or_no_status"

    finished = (df["status"].isin(["Finished", "Finish"])).any()
    if not finished:
        last_status = df["status"].iloc[-1] if len(df) else "unknown"
        return DNF_SCORE, f"dnf:{last_status}"

    race_time_s = df["race_time_ms"].max() / 1000.0
    # NaN = no collision; count only rows where collision_type is a non-empty string
    if "collision_type" in df.columns:
        coll_series = df["collision_type"].fillna("")
        collision_count = int((coll_series != "").sum())
    else:
        collision_count = 0
    total_penalty = float(df["collision_penalty"].fillna(0).sum()) if "collision_penalty" in df.columns else 0.0
    final_soc = float(df["soc"].iloc[-1]) if "soc" in df.columns else 1.0

    # score = race time (s) + SOC penalty from collisions×10
    score = race_time_s + total_penalty * 10.0
    detail = f"time={race_time_s:.1f}s col={collision_count} penalty={total_penalty:.3f} soc={final_soc:.2f}"
    return score, detail


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "best_score": DNF_SCORE,
        "best_params": DEFAULT_PARAMS.copy(),
        "current_params": DEFAULT_PARAMS.copy(),
        "iteration": 0,
        "history": [],
    }


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def read_current_params() -> dict:
    """Read current parameter values from rule_based_input.py."""
    text = RULE_BASED_INPUT.read_text(encoding="utf-8")
    params = {}
    for name in PARAM_SPACE:
        m = re.search(rf"{re.escape(name)}=([0-9]+\.?[0-9]*)", text)
        if m:
            params[name] = float(m.group(1))
        else:
            params[name] = DEFAULT_PARAMS[name]
    return params


def write_params(params: dict) -> None:
    """Update DriverConfig parameter values in rule_based_input.py."""
    text = RULE_BASED_INPUT.read_text(encoding="utf-8")
    for name, value in params.items():
        text = re.sub(
            rf"({re.escape(name)}=)[0-9]+\.?[0-9]*",
            f"{name}={value:.2f}",
            text,
        )
    RULE_BASED_INPUT.write_text(text, encoding="utf-8")


def perturb(params: dict) -> tuple[dict, str, float, float]:
    """Perturb one randomly chosen parameter by one step."""
    name = random.choice(list(PARAM_SPACE.keys()))
    lo, hi, step = PARAM_SPACE[name]
    direction = random.choice([-1, +1])
    old_val = params[name]
    new_val = round(max(lo, min(hi, old_val + direction * step)), 3)
    new_params = params.copy()
    new_params[name] = new_val
    return new_params, name, old_val, new_val


def main() -> None:
    state = load_state()
    state["iteration"] += 1
    it = state["iteration"]

    print(f"\n[Optimizer] ========== Iteration {it} ==========")

    # --- Evaluate last run ---
    run_dir = find_latest_run()
    evaluated = False
    if run_dir:
        score, detail = compute_score(run_dir)
        evaluated = True
        current_params = state["current_params"]

        print(f"[Optimizer] Last run : {run_dir.name}")
        print(f"[Optimizer] Score    : {score:.2f}  ({detail})")
        print(f"[Optimizer] Params   : {current_params}")

        if score < state["best_score"]:
            print(f"[Optimizer] *** IMPROVED: {state['best_score']:.2f} -> {score:.2f} ***")
            state["best_score"] = score
            state["best_params"] = current_params.copy()
        else:
            print(f"[Optimizer] No improvement (best={state['best_score']:.2f}). Reverting to best.")

        state["history"].append({
            "iteration": it,
            "run": run_dir.name,
            "score": score,
            "detail": detail,
            "params": current_params.copy(),
        })
        # Keep history to last 100 entries
        state["history"] = state["history"][-100:]
    else:
        print("[Optimizer] No run data found yet. Using default params.")

    # --- Perturb from best params ---
    new_params, name, old_val, new_val = perturb(state["best_params"])
    state["current_params"] = new_params

    print(f"[Optimizer] Perturbing: {name} {old_val:.2f} -> {new_val:.2f}")
    print(f"[Optimizer] Next params: {new_params}")
    print(f"[Optimizer] Best so far: {state['best_params']}  score={state['best_score']:.2f}")

    write_params(new_params)
    save_state(state)
    print(f"[Optimizer] rule_based_input.py updated. State saved to {STATE_FILE.name}")


if __name__ == "__main__":
    main()
