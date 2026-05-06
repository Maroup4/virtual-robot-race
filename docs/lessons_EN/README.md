# aira — Step-by-Step Lessons

Welcome to the aira training curriculum.
These lessons take you from first setup to submitting your own algorithm in an official competition — step by step.

> **Platform**: [aira-race.com](https://aira-race.com) — sign up, join competitions, and track your ranking.

---

## Before You Start

**Target audience**
- Engineers, researchers, and students interested in AI, autonomous driving, or robot control
- No prior AI/ML experience required — basic Python and terminal skills assumed
- 18+ recommended (Google account required for some tools used in the lessons)

**Requirements**
- **Windows 10 or later** (Mac/Linux support is planned for a future release)
- Python 3.12+ — [Download here](https://www.python.org/downloads/)
- VSCode — [Download here](https://code.visualstudio.com/)
- Git — [Download here](https://git-scm.com/)
- A GitHub account — [Sign up here](https://github.com/signup)
- An aira account — [Sign up at aira-race.com](https://aira-race.com/login)

**Recommended VSCode extensions**
- **Python** (Microsoft) — Python language support
- **AI coding assistant** (Gemini Code Assist, Claude Code, Codex, etc.) — AI coding assistance used in lessons
- **Markdown Preview Enhanced** (shd101wyy) — preview these `.md` files with `Ctrl+Shift+V`; adds a copy button to every code block

---

## Lesson Index

Work through the lessons in order. Each one builds on the previous.

| # | Lesson | What you'll do | Time |
|---|--------|----------------|------|
| **00** | [Preparation](00_Preparation.md) | Fork the repo, set up Python + VSCode, run your first race | ~30 min |
| **01** | [Foundation](01_Foundation.md) | Understand the aira philosophy and the Decision-Making Cycle | ~15 min |
| **02** | [Live Q&A](02_Live_QA_NotebookLM.md) | Use NotebookLM as an AI Q&A assistant during development | ~15 min |
| **03** | [Manual Control](03_Manual_Control.md) | Drive manually with the keyboard, collect training data | ~30 min |
| **04** | [Log & Table Mode](04_Log_and_Table_Mode.md) | Replay recorded runs, analyse telemetry logs | ~30 min |
| **05** | [Rule-Based Control](05_Rule_Based_Control.md) | Write a lane-following algorithm, race autonomously | ~60 min |
| **06** | [AI Mode](06_AI_Mode.md) | Train a neural network by imitation learning, run inference | ~60 min |
| **07** | [How to Join the Race](07_How_to_Join_Race.md) | Configure `config.txt`, submit your result to the leaderboard | ~15 min |
| **08** | [Syncing with Upstream](08_Sync_with_Upstream.md) | Pull official updates into your fork without losing your work | ~15 min |
| **99** | [Glossary](99_Glossary.md) | Key terms: SOC, RACE_FLAG, WebSocket, torque, etc. | Reference |

**Total**: approximately 4–5 hours for the full curriculum (spread across multiple sessions is fine).

---

## How to Use These Lessons

1. **Open this folder in VSCode** — `File > Open Folder > virtual-robot-race`
2. **Open a lesson file** and press `Ctrl+Shift+V` to open the Markdown preview
3. **Copy code blocks** using the copy button that appears in the top-right of each block
4. **Paste into your terminal** by right-clicking (no `Ctrl+V` needed in the VSCode terminal)
5. **Ask questions** using an AI assistant (Gemini Code Assist, Claude Code, Codex, etc.) or NotebookLM (covered in Lesson 02)

> If you get stuck at any step, open a [GitHub Issue](https://github.com/aira-race/virtual-robot-race/issues) and describe what happened.

---

## After the Lessons

Once you've completed Lesson 07, you're ready to compete:

1. **Register** at [aira-race.com](https://aira-race.com) — set your display name (this becomes `NAME=` in `config.txt`)
2. **Join a competition** at [aira-race.com/competitions](https://aira-race.com/competitions) — note the Competition ID
3. **Set `COMPETITION_NAME`** in `config.txt` to the competition ID
4. **Set `RACE_FLAG=1`** and run `python main.py`
5. **Click POST** on the confirmation panel — your result is on the leaderboard

---

**Ready? Start with [Lesson 00 — Preparation](00_Preparation.md).**

---

← [Back to Docs](../README.md)
