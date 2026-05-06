# 3. Manual Control

In this lesson, you will drive the robot manually with your keyboard and try to complete the course.
Driving yourself will give you the intuition for "what it means to go fast" and "why mistakes happen" — a solid foundation for designing algorithms in later lessons.

> **▶ Watch this lesson first**: [Lesson 03 Manual Control (YouTube)](https://youtu.be/kwLAPdCD-jU?t=197)

---

## 1. Launch and Setup

### Configure the Launcher

Run `start.bat` to open the launcher. Set it up as follows:

| Setting | Value | Description |
|---------|-------|-------------|
| **Name** | Your name | Alphanumeric and underscore, up to 16 characters |
| **Competition** | Tutorial | Leave as-is |
| **Active** | `1` | Use Robot1 only |
| **R1 mode** | `keyboard` | Keyboard control mode |
| **Data save** | `ON` | Save driving data (used in later lessons) |
| **Race flag** | `TEST ONLY` | Practice run — not submitted to the leaderboard |

Once you've confirmed the settings, click **START**. Unity launches automatically and the race begins.

> **💡 Want to edit config.txt directly?** Everything set in the launcher is written to `config.txt` automatically. Advanced users can edit the file directly if they prefer. However, with the default `HEADLESS=0`, the launcher GUI will open and **whatever you click START with in the GUI will overwrite your edits**. To bypass the GUI entirely and use config.txt as-is, set `HEADLESS=1` before launching.

---

## 2. Basic Rules

> **▶ Video chapter**: [Basic Rules Explained](https://youtu.be/kwLAPdCD-jU?t=197)

- **Start:** Three red signals light up, then all go out. The lights going out is the start signal.
- **False Start:** Moving before the start signal results in a false start and disqualification.
- **Laps:** Complete 2 laps of the course; your total time is what counts.
- **Wrong-way driving:** Driving the course in reverse will deduct 1 from your lap count.
- **Battery (SOC):** Driving consumes battery. When SOC reaches 0%, the robot can no longer move.
- **Collision:** Colliding with walls or other robots incurs a penalty that reduces SOC.
- **Course Out:** Falling off the course results in disqualification.
- **Time Up:** Failing to complete 2 laps within 90 seconds also results in disqualification.

---

## 3. Keyboard Controls

| Key | Action |
|-----|--------|
| `W` | Accelerate |
| `Z` | Reverse |
| `J` | Steer left |
| `L` | Steer right |
| `I` / `M` | Center steering instantly |

> Releasing `J` or `L` automatically returns the steering to center. `I` / `M` centers it instantly.

---

## 4. Reading the Screen

### Camera View
When two robots are active, the left side shows Robot1's camera and the right side shows Robot2's.

### Target Display
The circle on screen is the "target." It visualizes keyboard input (`W`, `Z`, `J`, `L`) as XY coordinates. With full forward torque and centered steering, the target moves to (0, 1).

> **❓ "What do the XY coordinates actually represent?"**
> The X-axis is the steering angle (−0.524 to +0.524 rad) and the Y-axis is drive torque (−1 to +1) — a direct visualization of the **continuous control signals** sent to Unity each tick. Watch the circle move as you press different keys to see exactly how your inputs translate to control values.

### Tail Lamp on the Robot
A visual interface that communicates your inputs to the robot behind you.

| Display | Meaning |
|---------|---------|
| **Color (Hue)** | Steering direction: red = left, green = straight, blue = right |
| **Light height** | Forward throttle strength (gauge extends from bottom to top) |
| **Blinking** | Reversing |

> **❓ "Is the tail lamp color intentionally designed as a visual feature for the AI?"**
> Yes, it's an intentional design. The following robot's camera captures the tail lamp, so an AI can learn to read "the car ahead is turning left/right" from the color as an image feature. In Task 4 at the bottom of this page, watch the AI's tail lamp — the color change is a real-time visualization of the AI's decision-making.

---

## 5. Training Tasks

Run `start.bat` each time and update the launcher settings before clicking **START**.

> **❓ "Does keyboard input produce bang-bang control — is that usable data for AI training?"**
> It's not pure bang-bang. Holding a key down increments torque/steering by a fixed step every 50 ms (`TORQUE_STEP=0.25`, `STEER_STEP=0.20`); releasing snaps it back to 0 immediately. There is no low-pass filter. The practical tip: **tap keys briefly and repeatedly** rather than holding them down. Jerky, coarse driving gets saved to disk, and Lesson 06's AI will learn to drive just as jerkily.

> **❓ "Does the 50 ms communication lag corrupt the training data?"**
> **Not in practice.** The WebSocket cycle is constant, so the lag between your input and Unity's response is always the same fixed delay. The saved data records the control command you sent and the sensor data you received in the same 50 ms cycle. Since the AI also runs at 50 ms during inference, the lag is symmetric between training and deployment — it effectively cancels out.

### Task 1: Complete 2 laps with Robot1
Launcher settings: **Active=`1`, R1 mode=`keyboard`, Data save=`ON`**

Keep trying until you finish 2 laps.

### Task 2: Complete 2 laps with Robot2
Launcher settings: **Active=`2`, R2 mode=`keyboard`**

Switch to Robot2 and drive the same course. Do you notice any difference in the feel?

### Task 3: Experience failures
Intentionally try out false starts, course outs, battery drain from collisions, and other disqualification scenarios. Get a feel for how penalties work.

### Task 4: Race alongside the AI
Launcher settings: **Active=`1,2`, R1 mode=`keyboard`, R2 mode=`ai`**

Chase the AI robot and observe where the gap opens up between you.

### Task 5: Time attack
Launcher settings: **Active=`1`, R1 mode=`keyboard`**

Go for the fastest 2-lap total time. Beat your personal best!

---

> **❓ Having trouble?**
> Paste your error message directly into [NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e) and ask for help.

---

⬅️ [Previous lesson: 02_Live_QA_NotebookLM.md (Live Q&A)](02_Live_QA_NotebookLM.md) ｜ ➡️ [Next lesson: 04_Log_and_Table_Mode.md (Log & Table Mode)](04_Log_and_Table_Mode.md)
