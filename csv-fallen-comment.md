# Bug Report: `metadata.csv` records `status=Finish` for runs that actually ended in `Fallen` (off-course)

> **対象 / Target**: aira Virtual Robot Race — simulator `aira_Beta_1.7.exe`
> **報告者 / Reporter**: fork user (Robot1, table mode trial)
> **日付 / Date**: 2026-06-06
> **添付 / Attachments**: 5 run folders `Robot1/training_data/run_20260606_091101 … run_20260606_091344`

---

## TL;DR（日本語）

テーブルモードで5回走行したところ、**画面上は5回ともコースアウト（Fallen）して落下**したにもかかわらず、`metadata.csv` の最終行の `status` 列が **すべて `Finish`** と記録されていました。
落下は `pos_y` 列で明確に確認できます（5本とも最終行 `pos_y ≈ -0.10〜-0.106`、仕様上 `pos_y < -0.1` は「コースアウト」判定）。
さらに調査した結果、**`Fallen` という status は全21走行分の `metadata.csv` に一度も出力されていません**。
終了時の status 判定／CSV書き出しが、実際の終了原因（Fallen）に関わらず `Finish` を書いてしまっているバグと考えられます。

## TL;DR (English)

In table mode, the robot **visibly fell off the course (Fallen) in all 5 runs**, yet every run's
`metadata.csv` terminal row records **`status=Finish`**. The fall is unambiguous in the `pos_y`
column (all 5 end at `pos_y ≈ -0.10 … -0.106`; per the docs, `pos_y < -0.1` = off-course).
Moreover, the value **`Fallen` is never written to `metadata.csv` in any of the 21 runs inspected**.
The end-of-run status determination / CSV export appears to write `Finish` regardless of the
actual termination cause.

---

## 1. Expected vs Actual

| | Expected | Actual |
|---|----------|--------|
| Robot drives off course (`pos_y < -0.1`) | terminal `status = Fallen` | terminal `status = Finish` |
| `error_code` on off-course end | error/eliminated indication | `999` (normal) |
| `Fallen` value appears in logs when a robot falls | yes | **never appears in any run** |

### Spec references (from this repo)
`docs/lessons_EN/04_Log_and_Table_Mode.md`:
- > `pos_y` | Y coordinate [m]. ... **Below `-0.1 m` is judged as off-course.**
- status table: `Fallen | Fell off the course → Eliminated`

So a run whose terminal `pos_y < -0.1` **must** be reported as `Fallen`, not `Finish`.

---

## 2. Evidence

### 2.1 The 5 affected runs (terminal row of `metadata.csv`)

| run folder | terminal `status` | terminal `pos_y` | `race_time_ms` | `error_code` |
|------------|-------------------|------------------|----------------|--------------|
| run_20260606_091101 | **Finish** | **-0.1056** | 22866 | 999 |
| run_20260606_091140 | **Finish** | **-0.1052** | 23606 | 999 |
| run_20260606_091221 | **Finish** | **-0.1009** | 24109 | 999 |
| run_20260606_091302 | **Finish** | **-0.1033** | 25073 | 999 |
| run_20260606_091344 | **Finish** | **-0.1038** | 12733 | 999 |

All five terminal rows have `pos_y < -0.1` (off-course) but are labeled `Finish`.
The lateral position at termination is also at the track edge (`pos_x ≈ ±1.0`), consistent with falling off.

Example raw terminal line (run_20260606_091101):
```
id,...,status,pos_z,pos_x,yaw,pos_y,error_code,collision_type,...
607,26854,22866,frame_000607.jpg,0.539,0.000,0.000,Finish,-1.113833,-0.983466,13.693,-0.105570,999,,0.0000,
```

### 2.2 Contrast: a legitimate finish
Reference run `run_20260506_165600` (a real 2-lap completion) ends **on course**:
- terminal `status = Finish`, terminal `pos_y = +0.0042` (never drops below -0.1 during the run).

→ A correct `Finish` has `pos_y ≈ 0`. A mislabeled `Finish` has `pos_y < -0.1`. The two are
distinguishable today only via `pos_y`, not via `status`.

### 2.3 `Fallen` is never emitted
Across **all 21** `metadata.csv` files inspected (16 rule-based + 5 table), the only `status`
values ever written are:
```
StartSequence, Lap0, Lap1, Finish
```
`Fallen` (and `Lap2`, `BatteryDepleted`, `FalseStart`, `ForceEnd`) never appear. This suggests
the terminal-status serialization does not cover the elimination cases — at minimum `Fallen`.

---

## 3. Reproduction

1. `config.txt`: `ACTIVE_ROBOTS=1`, `R1_MODE_NUM=2` (table), `DATA_SAVE=1`, `HEADLESS=1`.
2. Provide a `Robot1/table_input.csv` that drives the robot off the course. (Any open-loop
   command sequence that drifts off works; here it was an open-loop replay of a rule-based run.)
3. Run `python scripts/headless_loop.py 5` (or a single `python main.py`).
4. **Watch the simulator**: the robot visibly leaves the track and falls (Fallen).
5. Open the resulting `Robot1/training_data/run_*/metadata.csv` and inspect the last row:
   `status=Finish` while `pos_y < -0.1`.

(The 5 run folders attached to this report were produced exactly this way.)

---

## 4. Root-cause hypothesis (for the maintainer)

The runtime *does* detect the off-course/eliminated state during play (the robot stops and
falls), but the **end-of-run status that gets written to `metadata.csv` is hard-set to `Finish`**
(or derived from a "race ended" flag) without branching on the actual termination cause.
Because `Fallen` never appears in any CSV, the most likely locations are:
- the function that finalizes/flushes the last metadata row at run end, and/or
- the status state-machine → CSV mapping, where elimination states (`Fallen`,
  `BatteryDepleted`, `FalseStart`, `ForceEnd`) are not propagated to the exported `status`.

A quick guard that already has the needed signal: if terminal `pos_y < -0.1`, the status must be
`Fallen`, not `Finish`.

---

## 5. Impact

- **Automated scoring/optimization is corrupted.** `scripts/optimize_params.py` treats
  `status ∈ {Finish, Finished}` as success and uses `race_time` as the score (lower = better).
  A fallen run is mislabeled `Finish` with a *short* time (e.g. 12.7 s here), so a hill-climber /
  RL loop would rank a **crash as the best result**. Any auto-tuning built on `status` is invalid.
- **Misleading analysis & tutorials.** Learners following lessons 04 (table mode) and 05
  (rule-based) will see `Finish` for runs that actually fell off, contradicting the lesson text
  and hiding failures.
- **DNF detection breaks.** Code that relies on `status` to detect elimination cannot, because
  the elimination statuses are never written.

---

## 6. Suggested fix

1. Write the **true terminal status** to the final `metadata.csv` row: one of
   `Finish / Fallen / BatteryDepleted / FalseStart / ForceEnd`, matching the in-sim outcome.
2. Add a correctness guard: terminal `pos_y < -0.1` ⇒ `status = Fallen` (never `Finish`).
3. Verify the other elimination statuses are serialized as well (none currently appear in logs).
4. (Optional) Set a non-`999` `error_code` for eliminated runs so downstream tooling can branch
   on it without parsing `pos_y`.

---

## 7. Attached data

The following 5 run folders are attached for debugging (each contains `metadata.csv`, `images/`,
`output_video.mp4`):

```
Robot1/training_data/run_20260606_091101
Robot1/training_data/run_20260606_091140
Robot1/training_data/run_20260606_091221
Robot1/training_data/run_20260606_091302
Robot1/training_data/run_20260606_091344
```

Look at the **last row** of each `metadata.csv` (`status` vs `pos_y`) and the `output_video.mp4`
(the fall is visible).
