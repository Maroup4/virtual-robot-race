# 1. Foundation

## Introduction
Thank you for completing the environment setup in `00_Preparation.md`.
In this section, you will learn the philosophy and big picture that forms the "foundation" of your learning in aira.

The goal here is to understand **"what, why, and how you will learn in aira."**

## What is aira? — A Place to Compete on Decisions, Not Speed
aira (autonomous intelligence racing arena) is not just a racing game.
**"NOT ABOUT SPEED. ABOUT DECISIONS."**
This phrase embodies everything aira stands for.

The goal is not simply to achieve the fastest lap time. In a world of limited energy (SOC: State of Charge, battery level 0.0–1.0) and ever-changing conditions, you are asked to keep making **the best judgment (DECISION)**.

As an engineer, you will design the algorithm that makes these judgments and give intelligence to your virtual robot.

> **❓ "What does 'decision-making' mean in CS or control engineering terms?"**
> In this platform, "decision-making" means **selecting a control input (drive torque and steering angle) every 50 ms**. In optimal control terms, this corresponds to minimizing a cost function; in reinforcement learning terms, it corresponds to a policy in a Markov Decision Process (MDP). aira supports both approaches. In Lesson 05 you will explicitly design costs (SOC level, lane deviation) as rule-based control; in Lesson 06 you will experience behavior cloning (imitation learning) that learns a policy from driving data.

> **❓ "How physically accurate is the SOC consumption model?"**
> SOC is computed by the Unity simulator and sent to Python as a 0.0–1.0 float every 50 ms. It uses a **simplified consumption model** that does not include motor non-linearity or regenerative braking — intentionally kept simple so you can focus on algorithm design. Additionally, by design of the race format, **collisions with other robots also consume SOC as a penalty**. On the Python side, you access it as the `soc` variable; Lesson 05 shows how to use it to limit torque output.

## Why Learn with aira? — 3 Benefits
1.  **Develop practical problem-solving skills**
    You will cultivate decision-making skills that consider trade-offs under real-world constraints — energy management, course awareness, and interaction with other robots.

2.  **Experience the cycle of theory and practice**
    You will repeatedly go through the most important cycle in AI and software development: "design an algorithm, run it, analyze the results, and improve."

3.  **Explore a wide range of technologies**
    Beyond Python programming, you will encounter rule-based control, AI (machine learning), data analysis, simulation technology, and other diverse skill sets required of today's engineers.

## The Decision-Making Cycle — The Heart of Learning
Learning in aira progresses by repeating the following four phases. We call this the "Decision-Making Cycle."

1.  **① Design**
    In VS Code — your main workspace — you design and implement the algorithm that will be the robot's brain, using Python.

2.  **② Compete**
    The robot loaded with your algorithm races inside the Unity simulator. This is the moment your code becomes physical behavior in a virtual world.

3.  **③ Visualize**
    After a run, various log data is available. Why did it finish in that position? Where was energy wasted? The data vividly tells the story of the "decisions" your robot made.

    > **❓ "What data is actually available for visualization?"**
    > When `DATA_SAVE=1` is set, the following are saved automatically after each race:
    > - **`frames_map.csv`**: Per-tick (50 ms) drive torque, steering angle, SOC, and status flags
    > - **`metadata.csv`**: Unity's race log (session_time_ms, race_time_ms, SOC, control values, etc.)
    > - **Camera images (JPG)**: Front-view frame for every tick
    >
    > Lesson 04 covers how to load and plot these with `pandas` and `matplotlib`. Time-series analysis of control values is fully supported.

4.  **④ Improve**
    Based on insights gained from data analysis, you find the weaknesses and areas for improvement in your algorithm. Then you return to phase "① Design" and evolve your robot into a smarter one.

This cycle is the essence of engineering — and of AI development.

## What You Will Learn
In the upcoming training sessions, you will go through this cycle repeatedly while gradually making your robot smarter.

*   **Manual control**: First, experience firsthand what "the best decision" feels like as a human.
*   **Rule-based control**: Achieve basic autonomous driving through logical conditional branching.
*   **AI mode**: Implement an AI that makes more complex and adaptive decisions based on data.

> **❓ "Are rule-based and AI completely separate modes? Can I combine them?"**
> The modes are selectable, but the Python code is **fully composable**. For example, "switch to rule-based when SOC drops low" or "use AI inference only on fast corners" are all achievable by combining logic from `rule_based_input.py` and `inference_input.py`. Hybrid, layered architectures similar to real autonomous driving systems are very much welcome.

## Reflection

1.  **What do you expect from AI "judgment"?**
    In this training, you will learn both human-written logic (rule-based) and AI that learns from data. Even with rule-based control alone, the robot can drive — but what "judgments" would you want to leave to AI? Imagine a smarter way to drive that only AI can achieve.

2.  **Whose fault is it when something goes wrong?**
    The robot running your algorithm makes a strange move during a race (e.g., suddenly spins, runs out of energy before the goal) and finishes last. Whose responsibility is this "unexpected result"? And how would you use the "Decision-Making Cycle" to find the cause and apply it to the next race?

In the next section, you will learn how to use "NotebookLM," the AI chat that will powerfully support your learning.

---

⬅️ [Previous lesson: 00_Preparation.md (Preparation)](00_Preparation.md) ｜ ➡️ [Next lesson: 02_Live_QA_NotebookLM.md (Live Q&A)](02_Live_QA_NotebookLM.md)
