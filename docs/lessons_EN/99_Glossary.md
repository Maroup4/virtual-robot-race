# Glossary

This document summarizes the technical terms and important keywords used in the aira training project.

---

### A–Z

*   **ACTIVE_ROBOTS**: A setting in `config.txt`. Specifies the number of robots participating in the race (1 or 2).
*   **AI Assistant**: An AI such as Gemini Code Assist, Claude Code, or Codex that helps with coding. Active use of an AI assistant is recommended throughout this training.
*   **AI Mode**: A mode in which the robot drives autonomously based on inference results from a neural network model. Select **R mode=`ai`** in the launcher.
*   **aira**: Short for "autonomous intelligence racing arena." The name of this virtual robot racing platform.
*   **CNN (Convolutional Neural Network)**: A type of neural network commonly used to extract features from image data.
*   **config.txt**: The project-wide configuration file. Primarily used to specify the number of robots to run (`ACTIVE_ROBOTS`).
*   **CUDA**: A platform for parallel computing on NVIDIA GPUs. Used to accelerate AI training and inference.
*   **DAgger (Dataset Aggregation)**: A variant of imitation learning. Continuously adds the AI's own driving data to the training dataset to improve the ability to handle unknown situations.
*   **DATA_SAVE**: A setting in `config.txt`. When set to `1`, driving data (images, `metadata.csv`, etc.) is saved. When set to `0`, nothing is saved.
*   **drive_torque**: A variable that specifies the torque (rotational force) applied to the drive wheels. Ranges from -1.0 (maximum reverse) to +1.0 (maximum forward). One of the two fundamental control outputs.
*   **E2E (End-to-End)**: Meaning "from end to end." In AI mode, a method where a single neural network processes everything from image input to control output.
*   **Early stopping**: A technique in AI training that halts training early when performance on validation data stops improving, helping to prevent overfitting.
*   **Fork**: A GitHub feature that copies another user's repository to your own account.
*   **Git**: A version control system for recording and tracking changes to source code and other files.
*   **Google Colab**: A Google service for running Python code from a web browser. Provides free GPU access, which can dramatically reduce AI training time.
*   **GPU (Graphics Processing Unit)**: A chip originally designed for image processing. Capable of performing the massive parallel computations required for AI training at high speed.
*   **loss**: In AI training, a metric that represents the discrepancy (error) between the model's predictions and the ground-truth data. Training aims to minimize this loss.
*   **metadata.csv**: The most important data file at the center of a driving log. Every state of the robot (inputs, position, SOC, etc.) is recorded tick by tick.
*   **MIMO (Multiple Input, Multiple Output)**: A control system that determines multiple outputs from multiple inputs. In this project, the inputs are "image and SOC" and the outputs are "throttle and steering."
*   **MODE_NUM**: A setting in `config.txt`. Specifies the robot's control mode. `1`: keyboard, `2`: table, `3`: rule-based, `4`: AI, `5`: smartphone. Selectable via the **R1 mode / R2 mode** dropdown in the launcher.
*   **NAME**: A setting in `config.txt`. The player name displayed on the leaderboard.
*   **PyTorch**: An open-source machine learning library. Used primarily for AI model training and inference.
*   **RACE_FLAG**: A setting in `config.txt`. When set to `1`, the run result is registered on the leaderboard. When `0`, it is a test run only.
*   **SOC (State of Charge)**: Battery charge level. `1.0` = fully charged, `0.0` = empty.
*   **steer_angle**: A variable that specifies the steering angle of the front wheels. Unit is radians. One of the two fundamental control outputs.
*   **table_input.csv**: The CSV file used in Table Mode. Contains `drive_torque` and `steer_angle` per `time_id`; the robot drives exactly according to this file.
*   **tick**: The smallest unit of time in the simulation. In this project, one tick advances every 20 fps (50 ms).
*   **Unity**: A cross-platform game engine for creating 3D and 2D video games and interactive content. Used as the simulator in this project.
*   **Upstream**: In Git, the name of the remote pointing to the original (official) repository you forked from. Used to pull in updates from the original.
*   **Vibe Coding**: A development style where you interact with an AI assistant to program in a conversational or session-like manner — not just giving clear instructions, but also using the AI as a sounding board for ideas.
*   **VSCode (Visual Studio Code)**: A source code editor developed by Microsoft.

---

### Terms by Concept

*   **Architecture**: The overall structure and design philosophy of a system.
*   **Batch learning (Offline Learning)**: A method where training is performed using pre-collected driving log data (batch data), rather than learning in real-time during a run.
*   **BatteryDepleted**: A status value indicating the robot ran out of battery and can no longer move. It becomes an obstacle on the course.
*   **Course Out**: When the robot falls off the course, resulting in disqualification.
*   **DAgger (Dataset Aggregation)**: See A–Z section above.
*   **Distribution Shift**: When the distribution of data used during AI training differs from the distribution of data encountered during actual use (inference). In imitation learning, this can cause the AI to fail when it encounters situations the human never drove through.
*   **E2E (End-to-End)**: See A–Z section above.
*   **Epoch**: In AI training, the unit of one complete cycle through the entire dataset.
*   **Fallen**: A status value indicating the robot fell off the course and was eliminated.
*   **False Start**: The racing infraction of moving before the start signal. (See also: `FalseStart` status)
*   **Feedback Control**: A control method that observes the current state using sensors and corrects deviations from the target. Both rule-based control and AI control fall under this category.
*   **Finish**: A status value indicating the robot has completed the required number of laps.
*   **ForceEnd**: A status value indicating the race was forcefully terminated.
*   **Gain**: In control engineering, the ratio of how much the output changes relative to a change in input.
*   **Grid**: The starting position in a race.
*   **Hybrid Mode**: A control mode that combines multiple methods, such as using rule-based control for parts requiring reliability (like start detection) and AI for driving control.
*   **Imitation Learning**: A technique for achieving autonomous driving by training an AI on human driving data (demonstrations) so it mimics those inputs.
*   **Inference**: Using a trained AI model to make predictions or judgments on new data.
*   **Neural Network**: A mathematical model inspired by the neural circuits of the human brain. It is the "brain" of an AI.
*   **Offline Learning**: See Batch Learning above.
*   **Radian**: A unit of angle. Used for `steer_angle`. (π radians = 180 degrees)
*   **Reinforcement Learning**: A learning method where a robot learns through trial and error to take actions that yield better results (rewards). (**Note**: `rl_reward.py` in this project applies this concept for evaluating (scoring) driving quality from post-race data, not for real-time learning.)
*   **Robust**: The property of maintaining stable performance even when subjected to unexpected external influences (disturbances) or changes in environment. Also referred to as robustness.
*   **Rule-Based Control**: A method of controlling the robot based on rules defined by a human, such as "if the white line is to the right, turn the wheel left." Select **R mode=`rule_based`** in the launcher.
*   **Running**: A status value indicating the robot is currently driving.
*   **Sequential Control**: A control method that advances each stage of control in a predefined order. Table Mode is an example of this.
*   **Sliding Window Method**: An image processing technique. A small window is slid across the image to analyze information in each sub-region. Used for white line detection.
*   **StartSequence**: A status value indicating the robot is in the countdown to start.
*   **Tensor**: A multi-dimensional array. Used as the basic unit for handling numerical data in AI frameworks such as PyTorch.
*   **Time Up**: Failing to complete 2 laps within the allotted time (90 seconds), resulting in disqualification.
*   **Torque**: The force that rotates the tires. Controlled via `drive_torque`.
*   **Vibe Coding**: See A–Z section above.
*   **Wrong-way driving**: Driving the course in the opposite direction. Results in a lap count penalty.

---

⬅️ [Previous lesson: 07_How_to_Join_Race.md (How to Join a Race)](07_How_to_Join_Race.md) ｜ [Back to lesson list](README.md)
