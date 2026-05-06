# 4. Log Data Review and Table Mode

In this lesson, you will review the driving log (Log) saved during the previous manual control session, and learn about "Table Mode" — a mode that replays a run using that log data.

> **▶ Watch this lesson first**: [Lesson 04 Log Data & Table Mode (YouTube)](https://youtu.be/kwLAPdCD-jU?t=335)

**Learning goals:**
- Understand the folder structure and contents of log data saved with `DATA_SAVE=1`
- Learn the meaning of the detailed driving data in `metadata.csv`
- Learn how to drive the robot using Table Mode based on a CSV file
- Learn the basics of editing log data to create custom driving patterns

---

## 1. Reviewing Log Data

When you drive with `DATA_SAVE=1` set in the `03_Manual_Control` lesson, a folder named `run_[date]_[time]` is created inside `Robot1/training_data/`. This is a "log folder" containing all data from a single run.

### 1.1 Log Folder Structure
The log folder contains the following main files and folders:

- **`images/` folder**
  - All frames captured by the robot's camera during the run, saved as JPEG images.

- **`metadata.csv`**
  - The **most important data file** for a run. Every state of the robot during the run is recorded tick by tick (one tick = one simulation frame).

- **`output_video.mp4`**
  - A replay video automatically generated from the images in the `images/` folder.

- `UnityLog.txt`, `terminal_log.txt`
  - Detailed operation logs for debugging.
  - Contains logs from the Unity side and the Python side respectively.

---

### 1.2 Contents of `metadata.csv`
Open `metadata.csv` with a spreadsheet application like Excel or Google Sheets. The following **16 columns** appear from left to right.

#### Time / Identification

| Column | Contents |
|--------|----------|
| **`id`** | Sequential index for each tick. A unique number throughout the file. |
| **`session_time_ms`** | Elapsed time (ms) from the start of the start sequence. Counted even during the countdown. |
| **`race_time_ms`** | Elapsed time (ms) from the moment the "GO" signal fires. Stays `0` during the countdown. |
| **`filename`** | The corresponding image filename in the `images/` folder (e.g., `frame_000079.jpg`). |

> **💡 Data volume guide**: 1 tick = 50 ms, so **1,000 rows ≈ 50 seconds** of data. A 2-lap run will typically produce around 1,000–2,000 rows. The row count alone gives you a quick estimate of total run time.

#### Robot Operation / State

| Column | Contents |
|--------|----------|
| **`soc`** | Battery level (State of Charge). `1.0` = fully charged, `0.0` = empty. |
| **`drive_torque`** | Normalized drive torque. `+1.0` = maximum forward, `-1.0` = maximum reverse, `0.0` = stopped. |
| **`steer_angle`** | Front wheel steering angle (**in radians**). Range: **±0.524 rad (±30 degrees)**. Negative (`-`) = **left**, positive (`+`) = **right**. |
| **`status`** | A string representing the robot's state (see table below). |

**`status` value list:**

| Value | Meaning |
|-------|---------|
| `StartSequence` | Countdown to start |
| `Running` | Running |
| `Lap0` | Start of lap timing (used during training) |
| `Lap1` / `Lap2` ... | After passing a lap checkpoint |
| `Finish` | Completed the required number of laps |
| `FalseStart` | False start (moved before GO) → Disqualified |
| `Fallen` | Fell off the course → Eliminated |
| `BatteryDepleted` | Battery dead → Cannot move (becomes an obstacle) |
| `ForceEnd` | Forced termination |

#### Position / Orientation

> **Note**: The column order is `pos_z, pos_x, yaw, pos_y` — not alphabetical.
> **[Important]** `steer_angle` is in **radians**, but `yaw` is in **degrees**. When overlaying both on the same graph, convert units first:
> - **Excel / Google Sheets**: `=DEGREES(steer_angle)` converts to degrees
> - **Python (NumPy)**: `numpy.degrees(steer_angle)` or `steer_angle * (180 / numpy.pi)`

> **❓ "Why is `steer_angle` in radians but `yaw` in degrees?"**
> Unity manages Euler angles internally in degrees, so the world-space heading (`yaw`) is exported as-is in degrees. `steer_angle`, on the other hand, is used in the Python control model where radians are convenient for trigonometric functions (sin, cos), so it stays in radians. This mixed-unit design is a known gotcha — always convert to a common unit before analysis.

> **❓ "Writing unit conversion code every time violates DRY — can I build a wrapper class that normalizes units on load?"**
> Absolutely — and you should. It's exactly the right engineering instinct. The core conversions look like this:
> ```python
> df["steer_deg"] = np.degrees(df["steer_angle"])  # radians → degrees
> df = df[df["race_time_ms"] > 0]                  # drop countdown rows
> ```
> Ask your AI assistant (Gemini Code Assist, Claude Code, Codex, etc.): "Write a DataLoader class that reads metadata.csv, converts steer_angle to degrees, and returns only rows where race_time_ms > 0." It will generate reusable code you can drop into any analysis script.

| Column | Contents |
|--------|----------|
| **`pos_z`** | Z coordinate [m]. **Forward/backward** position (positive = course direction). |
| **`pos_x`** | X coordinate [m]. **Left/right** position (positive = right). |
| **`yaw`** | Yaw angle (heading direction) [**degrees**]. Forward direction = 0, clockwise = positive. |
| **`pos_y`** | Y coordinate [m]. **Up/down** position (positive = up). Below `-0.1 m` is judged as off-course. |

> **❓ "Are these local (robot-relative) coordinates or world coordinates?"**
> **World coordinates.** These are Unity world-space positions with the origin near the course start point. Unity uses a **left-handed coordinate system** (Z = forward, X = right, Y = up), and the data follows the same convention. The non-alphabetical column order `pos_z, pos_x, yaw, pos_y` reflects priority: Z (forward/backward) is the primary racing axis, X (left/right) is secondary.

#### Error / Collision Information

| Column | Contents |
|--------|----------|
| **`error_code`** | Numeric error code. `999` during normal driving. |
| **`collision_type`** | Collision type for this tick: `"wall"`, `"robot"`, `"both"`, or `""` (no collision). |
| **`collision_penalty`** | Collision penalty rate applied at this tick (normally `0.0`). Immediately deducted from battery. |
| **`collision_target`** | Name of the collision target (e.g., `Robot2`, `Wall`). Empty if no collision. |

This file lets you analyze in detail "at what moment, with what input, where was the robot, and was there a collision."

---

## 2. Reproducing a Run in Table Mode

Table Mode reads `drive_torque` and `steer_angle` values from top to bottom in `Robot1/table_input.csv` and drives the robot accordingly.
In other words, **the robot automatically drives according to a "blueprint" (table) you provide.**

### Format of `table_input.csv`
This file is very simple — it has only 3 columns:
- `time_id`: Sequential action number (integer starting from 0).
- `drive_torque`: Throttle input.
- `steer_angle`: Steering input.

---

## 3. Training Tasks

### Task 1: Reproduce a Run from Log Data

1.  **Prerequisites**
    - Run `start.bat` to open the launcher and set the following:

    | Setting | Value |
    |---------|-------|
    | **Active** | `1` |
    | **R1 mode** | `table` |
    | **Data save** | `OFF` |

    Click **START**.

2.  **Create table data**
    - Open the log folder created during the previous practice session, and open `metadata.csv` in a spreadsheet application.
    - Select and copy the data in the `drive_torque` and `steer_angle` columns, up to the row where you finished (e.g., up to row 1000).

3.  **Edit `table_input.csv`**
    - Open `Robot1/table_input.csv` in a text editor or spreadsheet application.
    - **Delete all the old data in `drive_torque` and `steer_angle` (columns B and C), keeping column A (`time_id`) intact.**
    - Paste the copied `drive_torque` and `steer_angle` data into columns B and C.
    - **Fill column A (`time_id`) with sequential integers starting from `0`.** In Excel, enter `0` and `1`, then drag the fill handle to extend the sequence.
    - Save the file in CSV format.

4.  **Check the run**
    - Run `start.bat`.
    - Confirm that the robot starts driving automatically according to the log you copied. Seeing your own keyboard inputs replayed may feel a little uncanny.

> **Think about it: Did it reproduce exactly?**
>
> Very few people will have replicated a clean lap. This is due to **latency**.
>
> The data you used as a log contains the command values the robot **actually received**. In Table Mode, that data is sent from top to bottom at **20 fps (50 ms intervals)**. The robot receives and acts on it.
>
> However, several "offsets" occur:
> - **Communication delay**: Transmission is not guaranteed to succeed, and packet arrival timing varies.
> - **Mechanical delay**: There is a slight time difference between instructing torque and the wheel actually rotating.
>
> This is the same in the real world. Simply "moving exactly as commanded" cannot handle external disturbances. This type of control is called **sequential control**.
>
> To move as intended, you need either carefully crafted command values with margin, or the **feedback control** you will learn next.

> **❓ "Does Unity execute commands the moment they arrive, rather than replaying them by timestamp?"**
> Correct. Unity applies received commands immediately in the current physics frame — there is no timestamp-based replay buffer or delay compensation. This means communication jitter (timing variation) directly translates to control error.

> **❓ "Why 50 ms (20 fps)? Can't the control loop go faster?"**
> It's a balance of several factors: ① WebSocket round-trip overhead (a few ms even on localhost), ② Python AI inference time (designed to fit within 50 ms on CPU), ③ Unity rendering load. Achieving sub-millisecond cycles would require shared memory or UDP instead of WebSocket — but **simplicity and accessibility** were prioritized over raw speed.

> **❓ "Table Mode is open-loop control, right? Is Lesson 05 going to be PID control?"**
> Exactly — Table Mode is pure **open-loop control**. It reads no sensor data, so initial position errors and jitter accumulate unchecked. In Lesson 05 you will implement **closed-loop (feedback) control**: the robot reads its camera image, detects the lane offset, and corrects steering accordingly. It's conceptually equivalent to the proportional (P) term of a PID controller.

> **❓ "Could we add a timestamp-based buffering queue on the Unity side to implement delay compensation?"**
> Technically yes. Implementing a timestamped command queue in Unity's C# and having Python attach a send-time to each command would enable more precise replay. However, there are practical constraints: ① localhost jitter is typically 1–5 ms, which has limited impact within a 50 ms cycle; ② Unity's physics engine update rate (default 50 fps) is itself a bottleneck; ③ the implementation complexity increases significantly. **Designing feedback control that is robust to disturbances** is a more practical and fundamentally sound approach than chasing perfect replay.

> **❓ "Could we use UDP or shared memory to push the control cycle up to 10 ms (100 fps)?"**
> At the transport layer, yes. Python's `asyncio + socket` (UDP) or `multiprocessing.shared_memory` can achieve sub-millisecond latency, and Unity's C# has `System.IO.MemoryMappedFiles` for the same. However: ① Unity's `Fixed Timestep` (default 50 fps) must also be tuned or the simulator won't keep up; ② UDP requires you to implement retransmission and ordering guarantees yourself, which WebSocket provides for free. This is a worthwhile research topic if you want to implement advanced algorithms like MPC.

### Task 2: Edit the Driving Data (Advanced)

- Rewrite all `steer_angle` values in a specific section of `table_input.csv` to `0`, save, and observe how the robot moves (it should drive in a straight line).
- Try rewriting the `drive_torque` values for just the first curve to half their original value to simulate braking into the corner.

As you can see, Table Mode is highly effective for creating and testing precise autonomous driving patterns without AI or complex programs.

### Task 3: Visualize `metadata.csv`

Log data is just numbers, but **graphing it makes the run "visible."**
Try using Excel, Google Sheets, or Python (`pandas` + `matplotlib`) — whatever works for you.

> **💡 Not sure how to write the Python code?** Ask your AI assistant (Gemini Code Assist, Claude Code, Codex, etc.):
> ```
> Read the metadata.csv file under Robot1/training_data/,
> then write Python code to plot a line chart with
> race_time_ms on the x-axis and drive_torque and steer_angle on the y-axis.
> ```
> Paste the generated code and run it directly.

---

#### Task 3-1: Draw the Driving Route (Scatter Plot)

Plot `pos_z` (horizontal axis) and `pos_x` (vertical axis) as a scatter plot.

> **Hint**: Set the X-axis to `pos_z` and the Y-axis to `pos_x`, and connect the data points with lines to get the driving trajectory.

- What shape of the course can you see?
- Where are the start and finish points?
- Are there sections where the trajectory drifts between laps?

---

#### Task 3-2: View Control Inputs Over Time (Line Chart)

Plot a line chart with `race_time_ms` on the horizontal axis and `drive_torque` and `steer_angle` on the vertical axis (overlapping both lines makes comparison easier).

> **Hint**: Rows where `race_time_ms` is `0` are during the countdown. Using only rows where the value is greater than `0` makes the actual race section easier to see.

- Where are the sections where you are pressing the throttle, and where are you easing off?
- When the steering angle is large, what is the relationship with `drive_torque`?
- Can you see a difference in control patterns between corners and straights?

---

#### Task 3-3: View Position and Heading Over Time (Line Chart)

Plot a line chart with `race_time_ms` on the horizontal axis and three lines: `pos_z`, `pos_x`, and `yaw`.

- Do `pos_z` and `pos_x` change at the same time, or do they alternate? (What does that mean?)
- Which sections of the scatter plot (Task 3-1) correspond to sharp changes in `yaw`?
- Compare the changes in `yaw` with the changes in `steer_angle` from Task 3-2. What relationship can you see?

---

#### Task 3-4: Going Further (Optional)

> **💡 3D driving trajectory**: Use `mpl_toolkits.mplot3d` to plot `pos_x`, `pos_z`, and `pos_y` in 3D — this reveals the course's physical structure (slopes, drops).
>
> **💡 Control input heatmap**: Create a 2D histogram (`plt.hist2d`) with `steer_angle` on the X-axis and `drive_torque` on the Y-axis to instantly visualize your driving habits (braking tendencies, steering bias). Ask your AI assistant: "Draw a 2D histogram of steer_angle vs drive_torque from my metadata.csv."

> **❓ "Can I build a Streamlit or Dash dashboard that auto-visualizes new log folders?"**
> Highly recommended — it's a high-leverage investment in your development efficiency. The basic architecture: use the `watchdog` library to monitor `Robot1/training_data/` for new `run_` folders, render 3D trajectories and heatmaps with `plotly`, and display them in a Streamlit browser app. Ask your AI assistant: "Write a Streamlit dashboard that watches Robot1/training_data/ and automatically displays the latest metadata.csv as interactive plots." This tool will pay dividends throughout Lesson 05 and beyond.

---

### Related Resources
- [03_Manual_Control.md](03_Manual_Control.md)
- [05_Rule_Based_Control.md](05_Rule_Based_Control.md)
- [Glossary](99_Glossary.md)

---

> **❓ Having trouble?**
> Paste your error message directly into [NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e) and ask for help.

---

⬅️ [Previous lesson: 03_Manual_Control.md (Manual Control)](03_Manual_Control.md) ｜ ➡️ [Next lesson: 05_Rule_Based_Control.md (Rule-Based Control)](05_Rule_Based_Control.md)
