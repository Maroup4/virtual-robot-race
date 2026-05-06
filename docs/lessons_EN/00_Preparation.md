# 0. Preparation

## Introduction
Welcome! In this training, you will learn how to develop AI and rule-based control programs on the virtual robot racing platform "aira."
In this first session, you will set up the development environment you need on your PC.

**Estimated time**: Approx. 10 minutes (excluding download time for Python libraries, etc.)

Work through each step carefully for a smooth start.

*   **What is "aira"?** → [Concept Introduction (aira-race.com)](https://aira-race.com/getting-started)
*   **Video guide for environment setup** → [aira-race.com/getting-started](https://aira-race.com/getting-started)

aira: autonomous intelligence racing arena

---

## Target Audience

This training is intended for:

*   People interested in AI, autonomous driving, and robot control who want to learn hands-on — university students (3rd year or above) and working professionals.
*   No prior Python programming experience required. However, basic PC skills (typing, copy & paste, running commands in a terminal) are assumed.
*   People who agree to the terms of service for AI services (Gemini Code Assist, Claude Code, Codex, etc.) used in this training and can use them with their own accounts (18 years or older recommended).

---

## 1. Development Environment Requirements

First, check the hardware and software requirements for this training.

### Hardware Requirements

| Item | Required | Recommended | Notes |
|------|----------|-------------|-------|
| **OS** | Windows 11 | - | Mac/Linux not supported |
| **RAM** | 8 GB | 16 GB | For running AI (torch) + Unity simultaneously |
| **Disk space** | 5 GB | 10 GB | .venv ~2.5 GB + Unity ~1 GB + driving data |
| **GPU** | Not required | NVIDIA GeForce (CUDA-compatible) | GPU recommended for AI training |
| **Network** | **Required** | - | Needed to install libraries (~2.5 GB) |

> **❓ "Why is it Windows 11 only?"**
> The simulator is a Unity-built Windows executable (`.exe`). While Unity can technically target Mac and Linux from the same source, the current version of aira is provided for Windows only. Running a GUI app via WSL2 or Docker is technically possible, but requires complex GPU, networking, and window-rendering configuration, so it is not officially supported.
> **Mac / Linux users**: The AI training portion (Lesson 06) can be experienced in **Google Colab** (browser-based, no simulator needed). See the `colab/` folder for details.

### Software Requirements

| Software | Version | Source | Installation Notes |
|:---------|:--------|:-------|:------------------|
| **Python** | 3.12 or later (64-bit) | [Official site](https://www.python.org/downloads/) | **Be sure to check "Add Python to PATH"** |
| **VSCode** | Latest | [Official site](https://code.visualstudio.com/) | |
| **Git** | Latest | [Official site](https://git-scm.com/) | Default settings are fine |
| **Google account** | - | [Sign-up page](https://accounts.google.com/signup) | Used for NotebookLM and Gemini Code Assist (if using Gemini) |
| **GitHub account** | - | [Sign-up page](https://github.com/signup) | Required to fork the repository |
| **PayPal account** | - (Optional) | [Sign-up page](https://www.paypal.com/signup) | Required to receive prize money in paid competitions |

> **⚠️ Most Important**
> If you forget to check "**Add Python to PATH**" during installation, the `python` command will not be recognized and all subsequent steps will fail. If you missed it, uninstall Python and reinstall it.

---

## 2. Setup Steps

Here are the actual steps. You will build your own personal development environment on your PC.

> **▶ Watch this section in the video**: [Setup Steps (YouTube)](https://youtu.be/kwLAPdCD-jU)

### Step 1: Prepare Your Own Repository (Fork & Clone)

"**Fork**" means creating your own personal copy of the official repository on GitHub.
"**Clone**" means downloading that copy to your PC.

We'll use **GitHub Desktop** (a free app) to do both. No terminal commands needed.

1. **Install GitHub Desktop**
    - If you haven't already, download and install it first.
    - → [Download GitHub Desktop](https://desktop.github.com/)
    - After installing, sign in with your GitHub account.

2. **Fork the official repository**
    - Open the [official repository](https://github.com/aira-race/virtual-robot-race) in your browser.
    - Click the **Fork** button in the upper right.
    - Click **"Create fork"** to finish.
    - You now have your own repository at `https://github.com/YOUR_USERNAME/virtual-robot-race`.

3. **Clone with GitHub Desktop**
    - After forking, click the green **Code** button on the repository page.
    - Select **"Open with GitHub Desktop"**.
    - GitHub Desktop will open — confirm the save location and click **Clone**.
    - Once the `virtual-robot-race` folder is created on your PC, you're done.

> **💡 Prefer the terminal?** You can do the same with:
> ```bash
> git clone https://github.com/YOUR_USERNAME/virtual-robot-race.git
> ```

> **💡 About future updates**: When the official aira repository is updated, there is a way to pull those changes into your own fork. This is called "syncing with Upstream." You don't need to worry about it now — it's covered in detail in [Lesson 08: Syncing with Upstream](08_Sync_with_Upstream.md).

### Step 2: Set Up the Python Environment

Next, set up the Python development environment for controlling the robot.

1.  Run the setup script `setup_env.bat`.
    ```bash
    .\setup_env.bat
    ```
    > **💡 Why `.\`?**: In PowerShell, typing `setup_env.bat` alone is not recognized as a command. The `.\` prefix means "**the file in the current folder**" — it tells PowerShell to run `setup_env.bat` from the current directory.

    This script automatically creates a virtual environment, installs the required Python libraries, and generates a VS Code settings file.

    > **❓ "What does setup_env.bat actually install?"**
    > Internally, it runs `pip install -r requirements.txt`. The main libraries are:
    >
    > | Category | Key libraries |
    > |----------|--------------|
    > | Deep Learning | `torch`, `torchvision` (PyTorch) |
    > | Numerical / Data | `numpy`, `pandas` |
    > | Image Processing | `opencv-python`, `pillow` |
    > | Networking | `websockets`, `aiohttp` |
    > | Visualization | `matplotlib` |
    > | Other | `keyboard`, `qrcode`, `pyyaml`, etc. |
    >
    > See `requirements.txt` for the full list. Total download size is approximately 2.5 GB (PyTorch accounts for most of it).

    When the script completes, a message will guide you to the next steps. If you see `(.venv)` at the beginning of the prompt, it worked.

    > **💡 Want to work in VS Code's terminal?** Run the following in VS Code's terminal (`Ctrl + Shift + @`):
    > ```bash
    > .venv\Scripts\activate
    > ```
    > Once `(.venv)` appears in the prompt, you're ready.

    > **❓ "Why `.venv` instead of Anaconda or Docker?"**
    > Standard `venv` was chosen because it requires **no additional installation and minimal configuration**. Anaconda is powerful but its installer is several GB and can overwrite your system Python by default. Docker is great for portability, but displaying GUI apps (like the Unity simulator) requires extra setup. With `venv + pip`, it works as long as Python is installed, and cleanup is as simple as deleting the `.venv` folder.

### Step 3: Configure VS Code

Finally, configure `Visual Studio Code`.

1.  **Open the project folder**
    - Launch VS Code, go to "File" > "Open Folder", and select the `virtual-robot-race` folder.

2.  **Install recommended extensions**
    - Search for and install the following extensions in the VS Code extensions marketplace:
      - **Python** (by Microsoft): Essential tool for Python development.
      - **AI coding assistant** (Gemini Code Assist, Claude Code, Codex, etc.): AI-powered coding assistance. You will use this in the latter half of the lessons. Install whichever you prefer. (Gemini Code Assist requires a Google account.)
      - **Markdown Preview Enhanced** (by shd101wyy): Makes reading these lesson documents much easier. With a `.md` file open, press `Ctrl + Shift + V` to open the preview — **a copy button will appear in the top-right corner of every code block.**

    > **💡 Pasting into the terminal**: In the VS Code terminal, simply **right-click** to paste — no need for `Ctrl+V`.

3.  **Verify the Python interpreter (usually not needed)**
    - `setup_env.bat` automatically generates a VS Code settings file, so `.venv` should be recognized as soon as you open the project folder.
    - If it is not recognized, open the command palette with `Ctrl + Shift + P`, type `Python: Select Interpreter`, and manually select `./.venv/Scripts/python.exe`.

---

## 3. GPU Acceleration (Optional)
If your PC has an NVIDIA GPU, you can significantly speed up **AI training (Lesson 06)**.

> **❓ "Do I need a GPU just to run the robot (inference)?"**
> **No.** CPU alone handles real-time inference at the 50 ms control cycle without issues. GPU acceleration mainly benefits the **training phase** (updating the model from large amounts of driving data). Feel free to skip this section and come back when training speed becomes a concern in Lesson 06.

1.  Open the virtual environment terminal (the one with `(.venv)` at the start of the prompt).
2.  Run the following two commands **in order**. The first removes the CPU version; the second installs the GPU version.
    ```bash
    pip uninstall torch torchvision -y
    ```
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```
3.  Check whether the GPU is recognized.
    ```bash
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```
    If you see `CUDA available: True`, you are done.

---

## 4. Verification

Let's do a test run to confirm everything is set up correctly.

1.  Open a new terminal in VS Code (`Ctrl + Shift + @`). Confirm that `(.venv)` is shown in the prompt.
2.  Run `start.bat`.
    ```bash
    .\start.bat
    ```
    > **💡 `python main.py` works too.** `start.bat` is a shortcut that activates `.venv` then runs `python main.py`. If `.venv` is already active, either works.

    > **❓ "How do Python and Unity communicate?"**
    > They communicate over **WebSocket** (port 12346). Unity acts as the server and Python connects as a client, exchanging sensor data (camera image, speed, etc.) and control commands (steering, throttle) at approximately 50 ms intervals. No standard interface like OpenAI Gym is used — aira uses its own simple protocol. You will read the actual code in **Lesson 05 (Rule-Based Control)**.
3.  Confirm the following:
    > **💡 First launch**: A Windows Security (firewall) dialog may appear. Click **"Allow access"** to continue. This is needed for communication between Unity and Python.

    - [ ] The launcher window opens. **Enter your name** and click **START**.
    - [ ] The Unity simulator **launches automatically**.
    - [ ] Two robots start driving the course automatically (default: R1 = rule-based, R2 = AI).
    - [ ] After the race ends, the Unity window closes automatically.

    > **💡 Data saving**: By default, `Data save` is `OFF`. To save driving data, set it to `ON` in the launcher (covered in Lesson 4).

Setup is complete! Well done.

> **❓ Having trouble?**
> Ask the AI mentor at [Lesson 02: Live Q&A (NotebookLM)](02_Live_QA_NotebookLM.md). Paste your error message and it will help you troubleshoot.

---

➡️ [Next lesson: 01_Foundation.md (Foundations)](01_Foundation.md)
