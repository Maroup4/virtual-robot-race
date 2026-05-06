# 5. Rule-Based Control and AI Orchestration

In this lesson, you will tackle rule-based control — which may sound difficult.
However, you do not need to write Python programs from scratch. **The essence of this lesson is to use AI assistants (such as Gemini Code Assist, Claude Code, or Codex) to the fullest to complete the tasks.**

**Learning goals:**
- Understand "what" this system controls (inputs, outputs, timing)
- Experience developing and improving control algorithms through Vibe Coding with an AI assistant
- Implement your own "winning strategy" and actually race with it

> **▶ Watch this lesson first**: [Lesson 05 Hands-on (YouTube)](https://youtu.be/kwLAPdCD-jU?t=464)
> The video has chapters. Use the links in each section to jump directly to the relevant part.

---

## 0. The Philosophy of This Lesson: Becoming a "Designer Who Moves Forward with AI"

This lesson is not for people who can already write Python.

Its purpose is to develop people who can **move forward by building code together with an AI assistant**.

---

### Designers over coders

Traditional programming education aims to "be able to write code from scratch yourself." That is genuinely a valuable skill.

But what this platform aims for is slightly different.

What is asked of you is the ability to **decide "what to build," delegate "how to build it" to AI, and evaluate and improve the results**.

This is not about "taking shortcuts." What the AI-era engineer truly needs is not the ability to write code line by line, but the ability to **design systems, realize them with AI, and correctly judge the results**.

---

### There's no reason not to use AI

Of course, it is possible to work through this lesson without an AI assistant.

However, reading through hundreds of lines of code, self-studying control theory, and manually tuning parameters takes enormous time. Using AI can dramatically shorten that process.

**There is no reason not to use it.**

---

### But don't let it be a black box

"The AI wrote it, so I don't need to understand the inside" — that won't work.

**A designer who cannot explain their own AI-generated code cannot improve it or debug it.** When the robot doesn't behave as expected, only you can judge what is wrong.

So in this lesson:
- Have the AI write the code, but **always have it explain what the code does**
- When changing parameters, **discuss the reasoning behind each value**
- When something goes wrong, **identify the cause together with the AI**

This mindset is the first step toward becoming an engineer who **truly co-develops with AI** — not just one who uses a black box.

---

### Your role in this lesson

| Role | Responsible party |
|------|------------------|
| Decide the strategy | **You** |
| Write and modify code | **AI assistant** |
| Evaluate the results | **You** |
| Propose the next improvement | **You + AI assistant** |
| Understand why, and **explain it in your own words** | **You** |

---

## 1. What Does This System Do? (MIMO Control)

### 1.1 Inputs and Outputs

This robot's control system has a structure called **MIMO (Multiple Input, Multiple Output)**.

| Type | Data | Details |
|------|------|---------|
| **Input ① Image** | RGB image | 224×224 pixel JPEG (from the robot's onboard camera) |
| **Input ② Battery** | SOC | Float value (e.g., `0.85`). Battery level from 0.0 to 1.0 |
| **Output ① Throttle** | `drive_torque` | Normalized torque from -1.0 to +1.0 |
| **Output ② Steering** | `steer_angle` | Steering angle ±0.524 rad (±30 degrees) |

From just **2 types of inputs**, the controller determines **2 types of outputs**.

> **🎯 This is an intentional constraint.**
> There is no speedometer. No GPS. No sensor that tells you where other robots are.
> All you are given is "the view (camera image)" and "remaining energy."
> Just as a human driver judges speed and steering using only their eyes, your algorithm **drives using vision alone.**
> This is the challenge — and the excitement — of this system.

### 1.2 What Must Be Achieved?

**Goal**: Complete 2 laps in the shortest time.

**Constraints:**

| Constraint | Details | Consequence if violated |
|------------|---------|------------------------|
| No false starts | Do not move before the GO signal | Immediate disqualification (FalseStart) |
| Stay on course | Do not fall off the course | Eliminated (Fallen) |
| Do not deplete battery | If SOC reaches 0, the robot cannot move | Becomes an obstacle (BatteryDepleted) |

> **Note**: Collisions with walls or other robots reduce SOC as a penalty, but do not cause immediate disqualification. However, they can be fatal if they accumulate.

**SOC consumption rules:**

SOC decreases from two sources:

| Source | Details |
|--------|---------|
| **Driving consumption** | Decreases in proportion to the absolute value of `drive_torque`. More throttle = more consumption. |
| **Collision penalty** | Additional SOC deducted each time you collide with a wall or another robot. |

In other words, "going faster = using more battery" is an inherent trade-off. Balancing speed and energy efficiency is one of the key design elements that determines the outcome.

### 1.3 Timing

The control loop runs at **20 fps (50 ms intervals)**. Every 50 ms, the image and SOC are acquired, an output is determined, and it is sent.

> **⚠️ What if processing exceeds 50 ms?**
> The system will not throw an error. Instead, **the previous frame's commands (throttle and steering) are held as-is**. Writing heavy processing means the robot keeps acting on stale instructions — and may drive straight into a corner wall. If you have the AI write complex code, always check its processing speed too.

---

## 2. File Structure and Data Flow

### 2.1 Main Structure

```
Robot1/
├── rule_based_input.py       ← Do not touch
├── rule_based_algorithms/    ← Your development zone (contents are fully yours)
│   └── ...
└── data_interactive/         ← Do not touch
    └── ...
```

### 2.2 Sample Algorithm Files

```
rule_based_algorithms/
├── sliding_windows.py          White line detection (sliding window method)
├── driver_model.py             Steering and speed decision logic
├── perception_Startsignal.py   Start signal detection (reads traffic light color from camera)
├── perception_Lane.py          Lane recognition utilities
├── perception_trackposition.py Track position estimation
├── status_Robot.py             Robot state management
├── Linetrace_white.py          White line tracing implementation example
└── debug_utils.py              Debug image generation (auto-saved to Robot1/debug/output/)
```

### 2.3 Basics

| File / Folder | Role | How to treat it |
|---------------|------|----------------|
| `rule_based_input.py` | Interface with main.py | No changes needed |
| `rule_based_algorithms/` | Algorithm development zone | Contents are completely yours |
| `data_interactive/` | Real-time data exchange location | Managed automatically by the system |

**Everything inside `rule_based_algorithms/` is yours to change.**
You can add files, rewrite existing ones completely, or replace the entire structure.

### 2.4 Why This Structure?

#### Role of `rule_based_input.py`

`main.py` calls only two functions from this file:

```python
soc = data_manager.get_latest_soc(robot_id)        # SOC: float 0.0 to 1.0
rgb_path = data_manager.get_latest_rgb_path(robot_id)  # Image path
pil_img = Image.open(rgb_path).convert("RGB")       # Load as PIL image
```

```python
def get_latest_command():
    return {
        "type": "control",
        "robot_id": robot_id,
        "driveTorque": round(float(driveTorque), 3),
        "steerAngle": round(float(steerAngle), 3),
    }
```

`update()` is called every 50 ms, runs your algorithm, and writes the result to global variables. `get_latest_command()` returns those values for WebSocket transmission. **Renaming either of these two functions will break the system.**

#### Role of `data_interactive/`

Unity writes images and SOC values here every 50 ms. `rule_based_input.py` only reads from it. Do not edit it directly — doing so will cause system malfunctions.

### 2.5 Recommended Reading Order

Trying to read all files at once leads to confusion. When having the AI assistant explain the code, the following order is most effective:

1. `rule_based_input.py` — The command center. Understand the overall flow first.
2. `sliding_windows.py` — Understand what is being read from the image.
3. `driver_model.py` — Understand how the readings are translated into throttle and steering.

Once you understand these three, the remaining files are "support components" you can read on demand.

### 2.6 Data Flow

```
Unity (Simulator)
  │  Updated every 50 ms
  ▼
data_interactive/
  latest_RGB_a or b  ← 224×224 JPEG
  latest_SOC.txt     ← Float text such as "0.850"
  │
  ▼ Read by rule_based_input.py
update() function
  ├─ Acquire and analyze image → compute lateral_px, theta_deg
  ├─ Acquire SOC
  ├─ driver_model.py determines drive_torque, steer_angle
  └─ Write to global variables driveTorque, steerAngle
  │
  ▼ Called by main.py
get_latest_command()  ← Output handling here
  └─ Returns {"type":"control", "driveTorque": ..., "steerAngle": ...}
  │
  ▼ Sent to Unity via WebSocket
```

### 2.3 Reading the Input

Inside the `update()` function in `rule_based_input.py`:

```python
soc = data_manager.get_latest_soc(robot_id)        # SOC: float 0.0 to 1.0
rgb_path = data_manager.get_latest_rgb_path(robot_id)  # Image path
pil_img = Image.open(rgb_path).convert("RGB")       # Load as PIL image
```

### 2.4 Writing the Output

Inside the `get_latest_command()` function in `rule_based_input.py`:

```python
def get_latest_command():
    return {
        "type": "control",
        "robot_id": robot_id,
        "driveTorque": round(float(driveTorque), 3),
        "steerAngle": round(float(steerAngle), 3),
    }
```

`update()` updates the global variables `driveTorque` / `steerAngle`, and `main.py` calls `get_latest_command()` to send them via WebSocket.

---

## 3. Run the Sample First
> **▶ Video chapter**: [Run the sample](https://youtu.be/kwLAPdCD-jU?t=464)

Run `start.bat` to open the launcher and set the following:

| Setting | Value |
|---------|-------|
| **Active** | `1` |
| **R1 mode** | `rule_based` |

Click **START** to run the robot.

> **Observation points:**
> - Did the robot drive the course autonomously?
> - Check the logs in the terminal (`Drive=`, `Steer=`, `LaneOK=`).
> - Look at the debug images saved in the `Robot1/debug/output/` folder. The white line detection is visualized there.

**Debugging tips:**

- **Print debugging**: You can write `print(variable_name)` anywhere inside `rule_based_algorithms/` and it will appear directly in the terminal. Use this aggressively whenever you want to inspect a value.

- **Toggle debug images on/off**: Controlled by the following line near the top of `rule_based_input.py`:

  ```python
  SAVE_DEBUG_OVERLAYS = True   # → Set to False to stop saving debug images
  ```

  Set it to `False` when racing or prioritizing processing speed. Debug images are saved every 50 ms, so long runs will fill up `Robot1/debug/output/` quickly. Get in the habit of clearing the folder periodically or switching back to `False` after testing.

---

## 4. Understanding the Algorithm (With an AI Agent)
> **▶ Video chapter**: [Reading code with an AI assistant](https://youtu.be/kwLAPdCD-jU?t=464)

Now the real work begins. Ask your AI assistant the following prompts in order to understand the system.

**Step 1: Check the current directory**
```
Can you access the current directory?
```

**Step 2: Understand the entire system**
```
Read the .md and .py files in the directory and understand what kind of system and application this is.
```

**Step 3: Analyze the rule-based algorithm**
```
Understand the contents of the "rule_based_algorithms" folder and explain what controls are implemented in the sample.
```

> **Tip**: The AI assistant will read the code and explain it. Even if you cannot read the code yourself, the AI will tell you in plain language "what it is doing."

**What to know when co-developing with AI:**

**Q: Does the AI know the course layout?**
No. The AI has no way to directly know the 3D shape or appearance of the course. However, by having it read the rules, constraints, and I/O specifications (such as `05_Rule_Based_Control.md`) and the code in `rule_based_algorithms/`, you can give it a reasonable understanding of "what environment it operates in." The key is to explicitly instruct: "Read these files, then propose something that accounts for the course constraints."

**Q: How do I catch the AI if it lies?**
If the AI writes a function that doesn't exist, running the code will produce an `ImportError` or `AttributeError`. When an error appears, **paste the error message directly back to the AI** and ask "I got this error — what's the cause and how do I fix it?" You can also ask the AI upfront: "Are all the functions and modules used in this code actually real?" before running anything.

**Q: Something isn't working. Is it the strategy or the code?**
Isolate the problem in steps:
1. Use `print()` to check intermediate values (`lateral_px`, `theta_deg`, etc.) → Are they what you expected?
2. Check debug images to confirm white lines are being detected correctly.
3. Once you identify the specific variable or behavior that's wrong, ask the AI: "This value looks off — why?"

"Something doesn't work" gives the AI nothing to go on. "This specific value is different from what I expected" produces dramatically better answers.

**Implementation Q&A:**

> **❓ "Can I replace sliding_windows.py with OpenCV Canny + HoughLinesP for better performance?"**
> Yes. `opencv-python` is already installed via `requirements.txt`. Understand the trade-offs before choosing:
> - **Sliding windows (current sample)**: Better at following curves. Easy to debug and interpret.
> - **Canny + HoughLinesP**: Faster straight-line detection. May struggle with the linear assumption on tight corners.
>
> The current sample already uses NumPy array operations — it is not doing pixel-by-pixel Python loops. Check the debug images (`SAVE_DEBUG_OVERLAYS=True`) to assess actual detection quality before optimizing.

> **❓ "Is the current driver_model.py just P control? Can I upgrade it to PID?"**
> Correct — it's P control. The line `steer = k_theta * theta_rad + k_lateral * lateral_n` is exactly two-input proportional control. Note that `alpha_smooth` is **output IIR smoothing** (`steer_s = (1-a)*steer + a*prev_steer`), not a D term on the error.
> To add PID: the I term accumulates `lateral_n` over time (**integrator anti-windup is essential**), and the D term is `lateral_n - prev_lateral_n`. Ask your AI assistant (Gemini Code Assist, Claude Code, Codex, etc.): "Add I and D terms to the DriverModel class in driver_model.py to make it a full PID controller. Include anti-windup protection."

> **❓ "Can I add EMA or a Kalman filter to handle temporary lane loss?"**
> The sample **already has a basic mechanism for this**. `rule_based_input.py` uses a `_lost_age` counter and `HOLD_FRAMES` (2 seconds' worth). When the lane is lost, the robot enters `hold` mode (maintaining the last command), then switches to `search` mode after 2 seconds. EMA can extend this by interpolating the detected lane position using a weighted average of recent frames. A Kalman filter is a more advanced approach that separates observation noise from prediction noise, smoothing out jittery detections.

> **💡 Prefer not to use an AI coding assistant?** [NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e) is also useful. Ask questions like "What does this file do?" or "Where does `lateral_px` come from?" and it will answer based on the actual source code.

---

## 5. Algorithm Development: Plan Your Winning Strategy
> **▶ Video chapter**: [Experience the algorithm development cycle](https://youtu.be/kwLAPdCD-jU?t=464)

### 5.1 Define Your Strategy

**You are now standing on the field of battle, together with an AI Agent.**

First, talk with the AI assistant to decide on a **strategy (policy)**.

> **💡 "Should I start with CV optimization or PID?"** Either is valid, but **strongly recommended: build the evaluation metrics in Task 4 first**. Without a baseline measurement, you can't tell whether your changes actually improved anything. Record your scores, then optimize.

```
Using this rule-based control system, propose a strategy for completing 2 laps in the shortest possible time.
Taking the winning conditions and constraints into account, tell me what parts of the current sample algorithm should be improved.
```

Example points to consider:
- How to go faster? (Raise `v_max`, adjust corner braking)
- Improve recovery when the lane is lost (`hold` / `search` logic)
- How to conserve SOC while driving? (Using `use_soc_scaling`)
- How to improve start signal detection accuracy?

### 5.2 Development Workflow

1. **Define policy**: Decide on an improvement policy through discussion with the AI assistant
2. **Implement**: Rewrite files in `rule_based_algorithms/` together with the AI assistant
3. **Test run**: Launch `start.bat` and actually drive the robot
4. **Analyze**: Check results using terminal logs, debug images, and `metadata.csv`
5. **Iterate**: Improve and race again

The sample program is just a starting point — feel free to use it as a base, but don't be constrained by its philosophy. Skillfully guiding the AI assistant is part of an engineer's role.

> **Tip**: Using the sample as-is is perfectly fine. Even changing a single parameter alters the driving behavior. Whether you make sweeping changes or fine-tune — the approach is entirely yours.

> **Tip**: It's fine to use the sample as-is. Even tweaking a single parameter changes the driving behavior. Whether you make sweeping changes or fine-tune — the approach is yours to choose.

### 5.3 Key Parameters (`driver_model.py`)

Here are some parameters you can easily change:

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `v_max` | Maximum speed (straight) | `0.55` |
| `v_min` | Minimum speed | `0.15` |
| `k_theta` | Gain on heading error | `0.45` |
| `k_lateral` | Gain on lateral offset | `0.30` |
| `alpha_smooth` | Output smoothing factor (larger = smoother) | `0.50` |
| `pulse_enabled` | Use pulse control in corners | `True` |
| `search_steer_const` | Steering angle when lane is lost [rad] | `0.6` |

### 5.4 Don't Forget Version Control

Now that control development has started, **commit and push to GitHub**.
Keeping a record of "versions you tried" lets you roll back when things go wrong.

```bash
git add Robot1/rule_based_algorithms/
git commit -m "Rule-based: v1.0 - First strategy"
git push
```

---

## 6. Training Tasks

### Task 1: Run the Sample and Change One Parameter

1. Set **R1 mode=`rule_based`** in the launcher and run, then record your time.
2. Change `v_max` in `driver_model.py` to `0.70` and run again.
3. What changed? Did it improve or worsen?

### Task 2: Improve with the AI Assistant

1. Share the result of Task 1 with the AI assistant and ask for the next improvement suggestion.
2. Implement the suggestion, run it, and compare.

### Task 3: Create Your Own "Winning Algorithm"

Create an algorithm that achieves the fastest time while respecting all constraints.
There is no single correct answer — experiment freely.

### Task 4: Build a Quantitative Evaluation Script (Advanced)

Move beyond "it feels smoother now." Build a script with your AI assistant (Gemini Code Assist, Claude Code, Codex, etc.) that automatically calculates the following metrics from `metadata.csv`:

```python
# Steering jitter (smoothness — lower is better)
jitter = df["steer_angle"].diff().dropna().std()
# Collision rate (0.0–1.0 — lower is better)
collision_rate = (df["collision_type"] != "").mean()
# SOC efficiency (final SOC / race duration — higher is more energy-efficient)
soc_efficiency = df["soc"].iloc[-1] / (df["race_time_ms"].iloc[-1] / 1000)
```

Ask your AI assistant: "Write a scoring script that reads metadata.csv, calculates jitter, collision rate, and SOC efficiency, and compares results across multiple run folders." This script will supercharge your improvement cycle in Tasks 1–3.

---

### Related Resources
- [04_Log_and_Table_Mode.md](04_Log_and_Table_Mode.md)
- [06_AI_Mode.md](06_AI_Mode.md)
- [Glossary](99_Glossary.md)

---

> **❓ Having trouble?**
> Paste your error message directly into [NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e) and ask for help.

---

⬅️ [Previous lesson: 04_Log_and_Table_Mode.md (Log & Table Mode)](04_Log_and_Table_Mode.md) ｜ ➡️ [Next lesson: 06_AI_Mode.md (AI Mode)](06_AI_Mode.md)
