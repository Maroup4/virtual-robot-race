# 6. AI Mode: Imitation Learning and Neural Networks

In this lesson, **you will train an AI to drive and let it run autonomously.**
It may sound difficult, but the essence of what you are doing is simple.
Make active use of an AI assistant (such as Gemini Code Assist, Claude Code, or Codex) here as well.

**Learning goals:**
- Understand where and how "inference" and "training" happen
- Experience the steps to create an AI model through imitation learning
- Feel the difference in training cost between local and Google Colab
- Understand the concept of reinforcement learning and its role in this system

---

## 1. First, Run the Robot in AI Mode

Run `start.bat` to open the launcher and set the following:

| Setting | Value |
|---------|-------|
| **Active** | `1` |
| **R1 mode** | `ai` |

Click **START**.

### 1.1 Prepare the Sample Model

AI mode requires a trained model file (`model.pth`).
The sample model is included in the repository — confirm that `Robot1/models/model.pth` exists.

### 1.2 Run It

Run `start.bat`. The following log will appear in the terminal:

```
[R1 Inference] Using device: cpu   (or cuda)
[R1 Inference] Model loaded from ...\Robot1\models\model.pth
[R1 Inference] Waiting for start signal... (Strategy: hybrid)
[R1 Inference] RACE STARTED! (Strategy: hybrid)
[R1 Inference] Drive=+0.523, Steer=-0.031rad(-1.8deg), SOC=1.00
```

> **Observation points:**
> - How did the robot drive?
> - The `Drive=` and `Steer=` values change every 20 fps (50 ms).
> - Depending on the model quality, it is normal if it does not drive well.

---

## 2. How Inference Works (`inference_input.py`)
> **▶ Video chapter**: [Understanding the inference pipeline](https://youtu.be/kwLAPdCD-jU?t=691)

Look at `Robot1/inference_input.py`. Ask your AI assistant:

```
What does inference_input.py do?
```

> **💡 No AI assistant? Use NotebookLM instead**: [Ask NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e)
> Paste the prompt above directly — it will explain the code based on the actual source files.

> **⚠️ Do not modify this file.** To customize AI behavior, edit `ai_control_strategy.py` instead (see Section 2.3).

---

### 2.1 The Core Inference Code

At the heart of the file are just a few lines of code:

```python
# Prepare input tensors
image_tensor = _transform(pil_img).unsqueeze(0).to(_device)  # [1, 3, 224, 224]
soc_tensor = torch.tensor([[soc]], dtype=torch.float32).to(_device)  # [1, 1]

# Run inference
with torch.no_grad():
    output = _model(image_tensor, soc_tensor)
    raw_drive = output[0, 0].item()   # drive_torque
    raw_steer = output[0, 1].item()   # steer_angle
```

**That is all.** Two pieces of data go in, two come out. This is MIMO.

---

### 2.2 Model Structure (`model.py`)

The neural network (`DrivingNetwork`) has a simple structure:

```
Input ①: RGB image 224×224      → CNN (4 conv layers + GlobalAvgPool) → 256-dim vector
                                                                          ↓ concat
Input ②: SOC (1 float)  ─────────────────────────────────────────────→ 257-dim vector
                                                                          ↓ MLP (fully connected)
Output: [drive_torque, steer_angle] (2-dim)
```

> **Note**: CNN (Convolutional Neural Network) extracts features from images.
> It converts visual information — like "the course curves right" or "the white line is shifted to the left" — into numbers.

> **Q: No speed sensor, single frame — is that enough?**
> The model has no direct way to measure speed. This is an intentional tradeoff. Using RNNs or stacked frames would make speed estimation easier, but it increases the amount of training data needed, adds implementation complexity, and tightens the 50 ms inference budget. Single-frame CNN was chosen for simplicity and speed. With enough training data, the model can learn appropriate torque/steering from the relative position of the white lines in the current frame.

---

### 2.3 Hybrid Mode vs. Pure E2E Mode

This system has two modes (configured in `ai_control_strategy.py`):

| Mode | Start Detection | Driving Control | Characteristics |
|------|----------------|----------------|----------------|
| **hybrid** (default) | Rule-based | AI | Start signal detection handled by reliable rules |
| **pure_e2e** | AI | AI | Fully delegated to AI (start signal is also learned) |

> There is no need to make AI learn things that can be reliably handled by rules.
> "Use rules for deterministic tasks, use AI for tasks that are hard to judge" is the practical approach.

**Advanced tuning parameters in `ai_control_strategy.py` (for engineers):**

Beyond the mode switch, the file exposes several fine-tuning knobs:

| Parameter | Effect |
|-----------|--------|
| `START_BOOST_FRAMES` | Duration of start-line boost (in frames) |
| `MIN_DRIVE_TORQUE` | Minimum torque applied during the boost phase |
| `STEER_SMOOTHING_ALPHA` | Low-pass filter coefficient for steering (0–1) |
| `CORNER_MIN_DRIVE_TORQUE` | Maximum torque cap during sharp corners |

If the AI drives erratically, adjusting these values often helps before retraining.

---

## 3. Training an AI Model

### 3.1 What Is "Imitation Learning"?

The learning method used in this system is **Imitation Learning**.

1. A human (you) drives using the keyboard
2. The data — "image + SOC → throttle + steering" — is recorded during that run
3. The AI learns the pattern: "A human would do this in response to this image"

In other words, **the AI learns to imitate your driving.**
If you want an AI that drives well, you need to give it data from good driving.

> **Important**: This training does not happen in real-time during a race.
> The flow is: collect driving data → **train offline** → run with the trained model.

---

### 3.2 Training-Related File Structure

```
Robot1/
├── model.py                     ← Neural network definition (do not modify)
├── models/
│   └── model.pth                ← The trained model to use (place it here)
│
├── training_data/               ← Data collected by keyboard driving (saved with DATA_SAVE=1)
│   ├── run_20260216_094415/
│   │   ├── images/
│   │   └── metadata.csv
│   └── run_.../
│
├── ai_training/                 ← Training scripts
│   ├── train.py                 ← Main training script
│   ├── run_scorer.py            ← Quality scoring for driving data
│   ├── create_iteration.py      ← Create iteration folder
│   └── analyze.py               ← Data analysis tool
│
└── experiments/                 ← Training results storage (.gitignore excluded)
    └── iteration_[timestamp]/   ← One training attempt
        ├── data_sources/        ← Copy of data used for training
        ├── model.pth            ← Model produced by this attempt
        ├── model_best.pth       ← Model with the lowest validation loss
        ├── training_log.csv     ← Loss per epoch
        └── dataset_manifest.json← Statistics of the dataset used
```

---

### 3.3 Local Training Steps

**Step 1: Collect data**

Run `start.bat` to open the launcher and set the following, then drive in keyboard mode.

| Setting | Value |
|---------|-------|
| **Active** | `1` |
| **R1 mode** | `keyboard` |
| **Data save** | `ON` |

After driving, a `Robot1/training_data/run_[datetime]/` folder is created.
At least 3 runs worth of data tend to produce better training results.

**Step 2: Run training**

Navigate to the `Robot1/` directory and run training:

```bash
cd Robot1
python ai_training/train.py --data training_data
```

> **💡 Start from a terminal open at the `virtual-robot-race/` root.** After `cd Robot1`, run `python ai_training/train.py ...`.

Training starts and loss is displayed per epoch:

```
Epoch   1/100 | Train: 0.045312 | Val: 0.048201 | LR: 1.00e-04 | 12.3s
Epoch   2/100 | Train: 0.038441 | Val: 0.041023 | LR: 1.00e-04 | 12.1s ✓ NEW BEST
...
⏹️  Early stopping triggered at epoch 47
Best validation loss: 0.012345
Model saved to: experiments/iteration_[timestamp]/model.pth
```

**Step 3: Place the model**

First, check the name of the folder that was created:

```bash
ls experiments/
```

You will see a folder like `iteration_20260502_143022`. Use that name in the following command (replace `iteration_20260502_143022` with your actual folder name):

```bash
copy experiments\iteration_20260502_143022\model_best.pth models\model.pth
```

> **PowerShell note**: Both `cp` and `copy` work as aliases. Avoid using the literal `[timestamp]` placeholder — PowerShell treats square brackets as wildcard characters, which will cause the command to fail.

> **Tip**: `model_best.pth` (the model with the lowest validation loss) tends to perform better than `model.pth` (the final epoch model).

**Step 4: Run in AI mode**

Set **R1 mode=`ai`** in the launcher and click **START**.

---

### 🎉 Congratulations — your AI just drove!

**Your robot just drove using an AI trained entirely on your own data.**

Take a moment to reflect:

> - **Did it drive at all?** Did it get off the start line?
> - **How far did it get?** First corner? One lap? Two laps?
> - **How did it look?** Wobbly? Crashing into walls? Leaving the course?
> - **Did it resemble your keyboard driving?** Can you spot your own habits in how it moves?

Whether it completed the course or crashed immediately, both outcomes are valid — they tell you exactly what your data can and cannot do yet.

---

### 3.3.5 Reading the Results: Why Did It Work — or Not?

Analyze your result from these angles:

**Common causes of poor performance:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Course-out immediately | Too little data (fewer than 3 runs) / rough keyboard inputs | Collect more runs with smoother inputs |
| Fails at the first corner | Too few examples of that corner | Drive that section more and re-collect |
| Wobbly straight-line driving | Jittery steering in training data | Practice smoother inputs before re-collecting |
| Never starts (stays still) | Start sequence not learned (pure_e2e only) | Switch back to hybrid mode |
| Falls apart on lap 2 | Training data was almost all lap-1 data | Collect complete 2-lap runs |

**Even if it worked, there is room to improve:**

- Time it. Is it faster or slower than your rule-based algorithm from [Lesson 05](05_Rule_Based_Control.md)?
- How much SOC is left at the end? Is energy being wasted?
- Plot `metadata.csv` from a run to see where energy is consumed (the techniques from [Lesson 04](04_Log_and_Table_Mode.md) apply here).
- Use `analyze.py` to visualize the torque and steering distribution across all your runs (run from `Robot1\`):

```bash
# Show torque and steering distribution graphs
python ai_training/analyze.py data

# Print a text summary of all runs
python ai_training/analyze.py summary
```

**Reading the output:**

| Column | Meaning | Good sign | Watch out for |
|--------|---------|-----------|---------------|
| `race_time_sec` | Lap time | Fast, consistent across runs | One run much slower than others |
| `torque_std` | Throttle variation | Low (smooth driving) | High (jerky input) |
| `steer_mean` | Average steering | Near 0 (balanced left/right) | Large offset to one side |
| `steer_std` | Steering variation | Low (stable control) | High (erratic steering) |

For example, a run with a much higher `steer_std` than the others likely contains erratic driving — the AI may learn that instability.

**Discuss with your AI assistant to decide the next move:**

```
I ran python ai_training/analyze.py summary and got this data:

[Training Data Report]
run_name             final_status  race_time_sec  torque_mean  torque_std  steer_mean  steer_std
run_20260502_063425  Finish        26.394         0.644        0.425       0.298       0.276
run_20260502_063547  Finish        22.315         0.757        0.374       0.319       0.275
run_20260502_063633  Finish        27.254         0.589        0.421       0.180       0.388

← (replace with your actual numbers)

Based on this:
1. Which runs have good quality and which are less suitable for training?
2. What kind of driving behavior do you predict the AI will learn from this data?
3. What should I do next to improve the model?
```

**The next step: collect more data and retrain**

If it crashed, resist the urge to blame the algorithm — add data and try again.

```
Improvement cycle:
  Observe AI → identify weak point → collect data for that situation → retrain → observe again
```

Focus your keyboard runs on whichever part of the course the AI struggled with, then re-run `python ai_training/train.py --data training_data`.

> **💡 This cycle is already DAgger in spirit (Section 6.2).** "Collect data for the situations the AI struggled with and retrain" — you have already started doing this instinctively.

---

> **⏱️ By the way — did local training take a long time?**
>
> On a PC without a GPU, training can take tens of minutes to over an hour.
> If you felt the urge to iterate faster, that feeling is the signal to move to the next section.

## 4. Training with Google Colab

### 4.1 Why Use Colab?

If your local PC has no GPU, training can take a very long time (tens of minutes to hours).
Using Google Colab's GPU, **the same data can train in just a few minutes to about 10 minutes**.

This is called "**training cost**." Experience the performance difference of a GPU and the meaning of paying for cloud compute.

---

### 4.2 Training Steps on Colab

The notebook `colab/train_on_colab.ipynb` is the step-by-step guide.
**Start by placing the notebook on Google Drive, then read and follow it from there.**

---

**Step 1: Create the `virtual-robot-race` folder in Google Drive**

Open a browser (Chrome recommended) and go to [drive.google.com](https://drive.google.com). Create a new folder directly under "My Drive".

> **⚠️ The folder name must be exactly `virtual-robot-race` (hyphen, no spaces).**

---

**Step 2: Upload the notebook and model.py**

Upload the following from your local PC to the folder you just created:

| Local file | Upload destination |
|-----------|-------------------|
| `colab/train_on_colab.ipynb` | `My Drive/virtual-robot-race/` |
| `Robot1/model.py` | `My Drive/virtual-robot-race/` |

---

**Step 3: Open the notebook in Colab**

In your browser, go to [drive.google.com](https://drive.google.com) and navigate to `My Drive/virtual-robot-race/`. **Double-click `train_on_colab.ipynb`** — Google Colaboratory will launch automatically and open the notebook.

> **💡 First time using Colab**: If double-clicking shows an "Open with" prompt, select "Google Colaboratory". If it is not listed, click "Connect more apps" to install it.

---

**Step 4: Switch the runtime to GPU**

In the Colab menu, go to **Runtime → Change runtime type**.

| Setting | Value | Notes |
|---------|-------|-------|
| **Runtime type** | Python 3 | Default — leave as-is |
| **Hardware accelerator** | **T4 GPU** | Available on the free tier; more than enough for this model |

Click **Save**.

> **💡 Which GPU to choose (as of May 2026):**
>
> | GPU | Cost | Relative speed | Recommended for aira |
> |-----|------|---------------|---------------------|
> | **T4 GPU** | Free (Colab free tier) | Baseline (~16 GB VRAM) | ✅ Start here |
> | L4 GPU | Paid (compute units) | ~2× T4 | — |
> | A100 GPU | Paid | ~5× T4 | — |
> | H100 GPU | Paid (expensive) | ~10× T4 | Overkill for this model size |
>
> aira's CNN model has roughly 4.5 million parameters — small enough that T4 completes training in a few minutes. H100 and A100 shine for large language models or deep ResNets, but for this model size the difference is negligible. **Start with T4 GPU (free).**

---

**Step 5: Run cells in order, following the notebook**

Press `Shift+Enter` to run each cell from Cell 1 onward. Cell 4 requires your training data to already be uploaded — the notebook will remind you at that point.

> **If English is a challenge**: return to `docs/lessons_JP/06_AI_Mode.md` Section 4 for the same explanation in Japanese.

---

**Step 6: Download model_best.pth and place it locally**

After training, download `experiments/iterations/iteration_[timestamp]/model_best.pth` from Google Drive.

**Recommended: rename the old model before replacing it**

```bash
cd Robot1\models

# Keep the current model under a new name so you can revert if needed
rename model.pth model_old.pth

# Copy the downloaded model_best.pth and rename it
copy C:\Users\YourUsername\Downloads\model_best.pth model.pth
```

> **💡 Why rename instead of overwrite?**
> If the new model drives worse than the previous one, you can go back to the old one by simply renaming `model_old.pth` back to `model.pth`. Overwriting leaves you with no way to compare or switch back.

> **Tip**: `model_best.pth` (lowest validation loss) tends to outperform `model.pth` (final epoch).

---

> **💡 Advanced: automate with sync_to_gdrive.py**
>
> If manually uploading run data every DAgger cycle becomes tedious, use the sync script (requires Google Drive desktop app):
>
> **First-time setup (creates folders and copies notebook automatically):**
> ```bash
> cd Robot1
> python scripts/sync_to_gdrive.py --setup
> ```
> **Upload new run data:**
> ```bash
> python scripts/sync_to_gdrive.py --sync-new
> ```


---

## 5. Batch Learning vs. Real-Time Learning — Where You Stand

You have now built an AI model from your own data and watched it drive.

Before moving on, let's take a moment to understand *what kind of learning you are actually doing* and *where it sits in the broader AI landscape*. Knowing this helps you understand why your model behaves the way it does — and what to try next.

---

**What you are doing right now: Imitation Learning (Batch Learning)**

```
Batch learning (imitation learning):  ← you are here
  Keyboard driving → data collection → offline training → place model → test run → repeat
```

"Batch learning" means you collect data first, then train — not while the robot is running. The model learns by copying your driving patterns, which is why it's also called *imitation learning*.

---

**How it differs from Reinforcement Learning**

```
Real-time reinforcement learning:
  While driving → compute reward → update model on the fly → keep driving
```

Reinforcement learning — the approach used in game-playing AIs and cutting-edge autonomous driving research — updates the model *while it acts*, rewarding good behavior and penalizing bad behavior in real time.

**aira uses batch learning by design — and that is intentional**

aira is built around the philosophy of *mastering fundamentals first*.

Real-time reinforcement learning is a powerful technique, but it comes with a steep learning curve: reward design, real-time sensing, training stability, and much more — all at once. Jumping straight there often means "things move, but you don't understand why."

**In aira, you first experience the most fundamental cycle of AI development: collect data, train, observe, improve.** The intuition you build here — *what data matters*, *why the model fails*, *what to change next* — is exactly the foundation needed when you move on to more complex systems: humanoid robots, autonomous vehicles, or any agent that must learn from the physical world.

> **The practical next step is DAgger (Section 6)** — a smarter way to iterate on imitation learning, and a natural bridge toward reinforcement learning thinking.

---

## 6. More Advanced Learning: DAgger and Reward Design

### 6.1 Limitations of Imitation Learning

The initially trained model has a weakness: when it encounters **situations the human never drove through**, it cannot handle them correctly.

For example:
- A situation where it drifts slightly off course → it doesn't know how to recover because there is no human data for it
- It gradually drifts further with each lap → even if the first lap is fine, it breaks down

This is called **"Distribution Shift."**

---

### 6.2 DAgger: Improve Through Iterative Data Collection

The technique that inspired this approach is **DAgger (Dataset Aggregation)**.

> **Clarifying the terms: Imitation Learning vs. Behavioral Cloning**
> - **Imitation Learning**: The broad category — any method that learns from human demonstrations
> - **Behavioral Cloning (bc)**: The simplest implementation — directly copies human actions using supervised learning
> - **DAgger**: A variant of imitation learning with a smarter data collection strategy
>
> So when you run `train.py --mode bc`, you are doing "Imitation Learning, specifically using the Behavioral Cloning method."

**Concrete example:**

```
[Start] Trained the initial model from 3 keyboard runs
  training_data/
    run_20260502_063425/   ← keyboard run ①
    run_20260502_063547/   ← keyboard run ②
    run_20260502_063633/   ← keyboard run ③

  → python ai_training/train.py --data training_data
  → copy model_best.pth → models/model.pth

[DAgger cycle 1] Run in AI mode (DATA_SAVE=ON) → run_04 is created
  training_data/
    run_20260502_063425/   ← keyboard run ①
    run_20260502_063547/   ← keyboard run ②
    run_20260502_063633/   ← keyboard run ③
    run_20260502_091012/   ← AI run ④  ★ newly added

  → python ai_training/train.py --data training_data
     ↑ uses runs 01–04 ALL TOGETHER (nothing is deleted)
  → copy new model_best.pth → models/model.pth

[DAgger cycle 2] Run again → run_05 is added
  → retrain on 01–05 → improves further
```

> **Q: Do I train only on the AI run (run_04)?**
> **A: No — use all of them together.** The keyboard data (01–03) stays in place; the AI data (04) is *added* to it. This is the "Aggregation" in DAgger. Discarding the human data removes the expert examples the model needs to learn from.

Each cycle adds data from situations the AI struggled with. Repeating it gradually broadens the model's coverage. This is the "Aggregation" in DAgger — always building on top, never discarding.

> **💡 Using Colab for training:** Let the AI drive locally → upload new data with `sync_to_gdrive.py --sync-new` → train on Colab → download model.pth → place locally. The same cycle applies.

> **Q: Won't my keyboard data get "diluted" as DAgger cycles pile up?**
>
> Two things to consider:
>
> **① AI data quality is poor** → `run_scorer.py` filters protect you. Fallen runs score around -500 and are auto-excluded by `min_score=0.0`. Tightening `--top-percent 30` keeps only the highest-scoring runs — which tends to be your cleaner keyboard data.
>
> **② AI data volume grows and keyboard proportion shrinks** → This is DAgger working as intended. As the AI improves, its runs approach (and can exceed) keyboard quality, so dilution stops being a problem. If AI quality is clearly worse than keyboard quality for many cycles, tighten `--top-percent` to prioritize the high-score runs.

> **⚠️ For those who want to dig deeper:** The DAgger paper requires an *expert* (human) to re-label actions for states visited by the AI. This system skips that step — the AI's own outputs are used as labels directly, and `run_scorer.py` filters out poor-quality runs (falls, false starts). This means incorrect behavior in novel states can still make it into training data. Tuning the score filter to use only completed runs helps reduce this.


---

### 6.3 Reward Design: Quantifying Data "Quality"
> **▶ Video chapter**: [Reward Design and Scoring](https://youtu.be/kwLAPdCD-jU?t=691)

Rather than using all data from DAgger equally, there is a mechanism to **weight "good runs" more heavily and "bad runs" less heavily (or exclude them)**.

This is handled by `rl_reward.py` and `run_scorer.py`.

| File | Role |
|------|------|
| `rl_reward.py` | Defines the evaluation criteria (reward weights) for a run's "quality" |
| `run_scorer.py` | Analyzes `metadata.csv` and assigns a score to each run |
| `train.py` | Selects and weights data based on scores for training |

> **Note**: `rl_reward.py` is **not** for updating the model in real-time.
> It is a "scoring criteria" for scoring driving quality by reading `metadata.csv` after the race ends.
> In the current version, position information (pos_x, pos_z) cannot be obtained in real-time, so performing real-time reinforcement learning during a race is not practical.

> **💡 When training on Colab:**
> The Colab notebook does not use `run_scorer.py` — it uses all uploaded run_ folders for training.
> To apply score-based filtering, **select your runs locally first, then upload**.
>
> ```bash
> # Check scores locally and identify the best runs
> python ai_training/run_scorer.py --top-percent 30
>
> # Upload only the runs you want to use
> python scripts/sync_to_gdrive.py --sync-new
> ```
>
> Since "what you upload to Google Drive = what Colab trains on," pre-filtering locally gives you the same effect.

---

### 6.4 What the Reward Evaluates

`rl_reward.py` and `run_scorer.py` currently implement the following evaluation criteria:

| Evaluation item | Content |
|-----------------|---------|
| Completion bonus | 2-lap finish (+1000), 1-lap finish (+400) |
| Time | Faster = higher score (baseline 120 sec, -2 per second) |
| SOC efficiency | Higher score for finishing with more battery remaining |
| Smoothness | Less steering jerk = higher score |
| Penalty | Fall (-500), forced termination (-100) |

By rewriting this "scoring criteria," you can change the definition of what constitutes "good driving."

**How scores are actually used in `train.py`:**

| Filter | Behavior | Default |
|--------|----------|---------|
| `filter_runs_by_score(min_score=0.0)` | Include only runs with score ≥ 0 | Falls (-500) and force-stops (-100) are auto-excluded |
| `get_top_runs(top_percent=50.0)` | Use only top N% of runs | Top 50% by default |

To inspect scores for your own runs (from `Robot1/`):

```bash
python ai_training/run_scorer.py
python ai_training/run_scorer.py --top-percent 30
```

Try one of these prompts with your AI assistant — pick the one that fits your situation:

**① Improve the scoring (basic)**
```
Look at the scoring criteria in run_scorer.py and rl_reward.py.
For the goal of "completing 2 laps in the shortest time,"
do you have ideas for more effective scoring?
Point out anything that looks off about the current weights.
```

**② Change the goal (prioritize battery efficiency)**
```
The current run_scorer.py evaluates both lap time and SOC efficiency,
but I want to prioritize "finishing with as much battery remaining as possible,
even if it means slower lap times."
Which parameters should I change, and how?
Please show me a code modification example.
```

**③ Debugging — AI keeps falling off**
```
I trained an AI model, but it keeps falling off at corners no matter how many times I run it.
Can I address this by changing the scoring or filtering settings in run_scorer.py?
Please show me how to reliably exclude fallen runs and retrain on completion-only data.
```

**④ Choosing a training mode (bc / rw / awr)**
```
train.py has three training modes: bc, rw, and awr.
I'm currently using bc mode with 5 runs of data.
Is it worth switching to rw or awr as a next step?
How much data and what completion rate would make the mode switch worthwhile?
```

**⑤ Add a new evaluation metric (custom reward)**
```
The current scoring doesn't evaluate "cornering stability."
Using the steer_angle and speed data in metadata.csv,
I want to add a new score item to run_scorer.py that measures
"whether the robot slowed down and steered smoothly through corners."
What would that implementation look like?
```

---

**💬 Example AI assistant exchange**

What kind of responses can you expect? Here are realistic examples based on the actual code values.

**For Prompt ① (basic scoring improvement):**

> **AI:** "I read `run_scorer.py`. The `finish_bonus` is `1000`, but `time_penalty_per_second` is only `2.0` — meaning being 60 seconds faster only adds +120 to the score. **Whether the robot finishes** dominates the score; lap time differences are nearly ignored. To make time matter more, try raising `time_penalty_per_second` to `4.0–6.0`. First, let's check the current score distribution."

```bash
python ai_training/run_scorer.py
```

```
Total runs: 5, Valid: 5
  1. [OK] run_20260502_063547:  1696.3 pts
  2. [OK] run_20260502_063425:  1690.0 pts
  3. [OK] run_20260502_063633:  1686.4 pts
  4. [NG] run_20260502_091012:  -482.1 pts  ← fallen
  5. [NG] run_20260502_094530:   -95.8 pts  ← force-stopped
Score range: -482.1 - 1696.3
```

> **AI:** "3 of 5 runs completed; 2 are falls/force-stops. With the default top-50% filter, the two negative-score runs may end up in training. Let's check what --top-percent 30 gives us."

```bash
python ai_training/run_scorer.py --top-percent 30
```

```
[Scorer] Filtered to top 30.0% (1 runs)
  1. [OK] run_20260502_063547:  1696.3 pts
```

> **AI:** "At 30%, only 1 run remains — too aggressive for your current data volume. I'd suggest keeping the default 50% for now and collecting more completion runs first. Once you have 8–10 completion runs, tightening to 30–40% will start to make a meaningful difference."

---

**For Prompt ③ (AI keeps falling off):**

> **AI:** "Repeated falls usually mean low-quality runs have leaked into training data. Let's check the scores first."

```bash
python ai_training/run_scorer.py
```

> **AI:** "Several runs are near -500. `train.py` already excludes falls with `min_score=0.0`, so those won't be used directly. But unstable runs in the 0–400 range — almost-fell-but-didn't — can still teach the model bad habits. Try filtering to the top 40% to focus on your cleaner runs."

```bash
python ai_training/run_scorer.py --top-percent 40
python ai_training/train.py --data training_data
```

> **AI:** "If you only have 2–3 completion runs, there may not be enough 'how to complete a lap' information in the data. The most effective fix is to add 3–5 more careful keyboard runs and retrain from those."

---

> **📝 Using Claude Code:**
> Claude Code (this assistant) can read files directly, so during exchanges like the ones above it will say things like "I checked `SCORING_WEIGHTS` in `run_scorer.py`" or "I confirmed the available `train.py` arguments" — giving you **concrete guidance grounded in your actual code values**. ChatGPT and Gemini can do the same, but you need to paste or upload the code first.

---

#### Advanced: AWR (Advantage-Weighted Regression) mode

`train.py` supports multiple training modes beyond standard behavioral cloning:

| Mode | Full name | What it does |
|------|-----------|-------------|
| `bc` (default) | Behavioral Cloning | The simplest form of imitation learning — all frames weighted equally |
| `rw` | Reward-Weighted BC | Uses run score as per-frame loss weight |
| `awr` | Advantage-Weighted Regression | Trains `ValueNetwork` to estimate V(s), weights by A(s,a) = Q(s,a) − V(s) |

**Command examples (run from `Robot1\`):**

```bash
# ① bc (default) — start here
python ai_training/train.py --data training_data

# ② rw — weight higher-scoring runs more heavily
python ai_training/train.py --data training_data --mode rw

# ③ rw + filter to top 50%
python ai_training/train.py --data training_data --mode rw --top-percent 50

# ④ awr — also trains ValueNetwork alongside the policy
python ai_training/train.py --data training_data --mode awr

# ⑤ awr + filter to top 50%
python ai_training/train.py --data training_data --mode awr --top-percent 50

# ⑥ fine-tune from an existing model (rw example)
python ai_training/train.py --data training_data --mode rw --finetune models/model.pth
```

**Typical progression:**

```
Step 1. Build a working model with bc first
  → python ai_training/train.py --data training_data

Step 2. Once you have 10+ runs, try rw
  → python ai_training/train.py --data training_data --mode rw --top-percent 50

Step 3. If rw plateaus, try awr
  → python ai_training/train.py --data training_data --mode awr --top-percent 50
```

> **💡 Customizing hyperparameters (advanced)**
>
> `train.py` supports `--batch-size`, `--lr`, and `--epochs` overrides:
> ```bash
> python ai_training/train.py --data training_data --batch-size 64 --lr 0.0005 --epochs 50
> ```
> For Colab (T4 GPU, 16 GB VRAM), you can increase the batch size by editing `BATCH_SIZE = 32` directly in Cell 6 of the notebook (128–256 is feasible). That said, for a model this size (~4.5M parameters), the benefit of a larger batch is limited. Start with the defaults first.

In AWR, the model learns to emphasize frames where the actual outcome was *better than expected* for that state. The `ValueNetwork` class in `model.py` is the value function implementation. Run scores from `run_scorer.py` are used as the reward signal — scoring and AWR are therefore closely connected.

> **Note**: AWR requires more data than `bc` and is more complex to tune. Build a working `bc` model first, then experiment with AWR.

Try one of these prompts with your AI assistant:

**① I have a working bc model — should I try rw next?**
```
Explain the difference between bc, rw, and awr modes in train.py.
I have a bc model that can complete laps, with 5 completion runs in my dataset.
What changes if I switch to rw mode?
Are there any downsides?
```

**② I want to understand what ValueNetwork does**
```
Look at the ValueNetwork class in model.py.
What does this network do?
Explain the difference between DrivingNetwork (policy) and ValueNetwork (value function)
using a concrete example from this robot racing context.
```

**③ AWR made my model worse than bc**
```
I trained with train.py --mode awr, but the model drives less stably than bc.
What are common causes and fixes?
My current dataset has 5 completion runs and 3 fallen runs.
```

**④ I want to compare bc, rw, and awr results**
```
I trained separate model_best.pth files with bc, rw, and awr modes.
How should I compare and evaluate the three models?
Is there a way to use run_scorer.py scores to quantitatively compare them?
```

---

## 7. Training Tasks

### Task 1: Build Your Own AI Model from Scratch

Go through everything you learned in this lesson — this time without following the steps in the text.

1. Configure the launcher and collect 3 runs of keyboard driving data
2. Run `train.py` locally to generate `model.pth` (**note the training time**)
3. Place it in `models/` and verify it drives in AI mode
4. Upload to Google Drive and run training on Colab as well
5. Compare local vs Colab training time — **how different was it?**

> **Check:**
> - Did your AI drive successfully? If it goes off course, which corner?
> - How much faster was Colab compared to local training?

### Task 2: Discuss a Strategy to Win with Your AI Assistant

Based on what you observed in Task 1, bring it to your AI assistant.

```
I built an AI model for a robot racing simulator.
I trained in bc mode using 3 runs of keyboard driving data,
and ran it in AI mode.

Current issues I noticed:
- [Write what you actually observed — e.g., "tends to go off course at right corners"]
- [e.g., "gradually drifts off in the second lap"]

Please prioritize what I should improve first to get a better finishing time.
Files available: train.py / run_scorer.py / rl_reward.py / ai_control_strategy.py
```

> **Tip:** Replace the "Current issues" section with what you actually observed. The more specific you are, the more useful the AI's advice will be.

---
### Related Resources
- [05_Rule_Based_Control.md](05_Rule_Based_Control.md)
- [04_Log_and_Table_Mode.md](04_Log_and_Table_Mode.md)
- [Glossary](99_Glossary.md)

---

> **❓ Having trouble?**
> Paste your error message directly into [NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e) and ask for help.

---

⬅️ [Previous lesson: 05_Rule_Based_Control.md (Rule-Based Control)](05_Rule_Based_Control.md) ｜ ➡️ [Next lesson: 07_How_to_Join_Race.md (How to Join a Race)](07_How_to_Join_Race.md)
