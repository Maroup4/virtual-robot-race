# 7. How to Join the Race

You have developed your algorithm and AI model through the training sessions — now it's time to race.
There are two race formats: "Time Attack" and "Head-to-Head."

**Learning objectives:**
- Complete the Time Attack race entry steps in practice
- Understand how results are automatically submitted to the leaderboard
- Understand the Head-to-Head format and how grid positions are determined
- Implement a robust start detection that works from either grid position

> **▶ Watch this lesson first**: [Lesson 07 Hands-on (YouTube)](https://youtu.be/kwLAPdCD-jU?t=1386)
> Chapters are set in the video. Use the section links to jump directly to each part.

---

## 1. Time Attack (Single Robot)

First, try the Time Attack where you drive solo and aim for the fastest time over 2 laps.

### Step 1: Finish Your Algorithm / AI Model
Complete the rule-based algorithm or AI model you developed during training.

### Step 2: Configure Race Settings
Run `start.bat` to open the launcher and set the following:

| Setting | Value | Description |
|---------|-------|-------------|
| **Name** | Your name | Alphanumeric and underscore, up to 16 characters. Shown on the leaderboard. |
| **Competition** | Competition ID | Enter the ID provided by the organizer. Use `Tutorial` for practice. |
| **Active** | `1` | Time Attack uses Robot1 only |
| **R1 mode** | `rule_based` or `ai` | Your algorithm |
| **Race flag** | `SUBMIT` | Submit result to the leaderboard |

> **Note (for competition mode)**: When entering a real competition ID, your `Name` must be pre-registered in the competition sheet. If not registered, an authentication error will appear before the race starts.

### Step 3: Drive
Click **START** to begin the Time Attack. When you finish, your result is **automatically submitted to the leaderboard**.

**Leaderboard:** [https://aira-race.com/](https://aira-race.com/)

---

## 2. Head-to-Head (Two Robots)

In the next training session, you will compete in direct head-to-head races.

### How to Submit
Check the active competition page for submission instructions and deadlines.

**Competition page:** [https://aira-race.com/competitions](https://aira-race.com/competitions)

### Race Format
- Submitted programs will race two at a time.
- Grid positions are determined based on Time Attack results. The faster driver starts as **Robot1** (front grid position).

> **[Important] Robustness of Start Control**
> Robot1 and Robot2 see the start signal from different positions and angles.
> The key to winning is implementing a **robust** start detection logic that fires reliably from either grid position.

---

Now, race with your algorithm!
**Race your algorithm!**

---

> **❓ Having trouble?**
> Paste your error message directly into [NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e) and ask for help.

---

⬅️ [Previous lesson: 06_AI_Mode.md (AI Mode)](06_AI_Mode.md) ｜ ➡️ [Next lesson: 08_Sync_with_Upstream.md (Syncing with Upstream)](08_Sync_with_Upstream.md)
