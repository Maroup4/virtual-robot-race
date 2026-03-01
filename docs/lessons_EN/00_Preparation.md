# 0. Preparation

## Introduction
Welcome! In this training, you will learn how to develop AI and rule-based control programs on the virtual robot racing platform "aira."
In this first session, you will set up the development environment you need on your PC.

**Estimated time**: Approx. 10 minutes (excluding download time for Python libraries, etc.)

Work through each step carefully for a smooth start.

*   **What is "aira"?** -> [Concept Introduction](https://www.youtube.com/watch?v=wAaaAODsfrE&t=26s)
*   **Video guide for environment setup** -> [YouTube Installation Guide](https://www.youtube.com/watch?v=cvUdITqjpc8)

aira: autonomous intelligence racing arena

---

## Target Audience

This training is intended for:

*   People interested in AI, autonomous driving, and robot control who want to learn hands-on — university students (3rd year or above) and working professionals.
*   No prior Python programming experience required. However, basic PC skills (typing, copy & paste, running commands in a terminal) are assumed.
*   People who agree to the terms of service for Google's AI services (Gemini Code Assist, etc.) used in this training and can use them with their own accounts (18 years or older recommended).

---

## 1. Development Environment Requirements — Beta 1.6
First, check the hardware and software requirements for this training.

### Hardware Requirements

| Item | Required | Recommended | Notes |
|------|----------|-------------|-------|
| **OS** | Windows 10 or later | Windows 11 | Mac/Linux not supported |
| **RAM** | 8 GB | 16 GB | For running AI (torch) + Unity simultaneously |
| **Disk space** | 5 GB | 10 GB | .venv ~2.5 GB + Unity ~1 GB + driving data |
| **GPU** | Not required | NVIDIA GeForce (CUDA-compatible) | GPU recommended for AI training |
| **Network** | **Required** | - | Needed to install libraries (~2.5 GB) |

### Software Requirements

| Software | Version | Source | Installation Notes |
|:---------|:--------|:-------|:------------------|
| **Python** | 3.12 or later (64-bit) | [Official site](https://www.python.org/downloads/) | **Be sure to check "Add Python to PATH"** |
| **VSCode** | Latest | [Official site](https://code.visualstudio.com/) | Japanese language pack recommended (if needed) |
| **Git** | Latest | [Official site](https://git-scm.com/) | Default settings are fine |
| **Google account** | - | [Sign-up page](https://accounts.google.com/signup) | Used for NotebookLM and Gemini Code Assist |
| **GitHub account** | - | [Sign-up page](https://github.com/signup) | Required to fork the repository |

> **⚠️ Most Important**
> If you forget to check "**Add Python to PATH**" during installation, the `python` command will not be recognized and all subsequent steps will fail. If you missed it, uninstall Python and reinstall it.

---

## 2. Setup Steps
Here are the actual steps. You will build your own personal development environment on your PC.

### Step 1: Prepare Your Own Repository (Fork & Clone)
First, create your own copy (fork) of the official repository, then download (clone) it to your PC.

1.  **Fork**
    - Open the [official repository](https://github.com/aira-race/virtual-robot-race) in your web browser.
    - Click the **Fork** button in the upper right to copy the repository to your GitHub account.
    - This creates your own repository at `https://github.com/YOUR_USERNAME/virtual-robot-race`.

2.  **Clone**
    - Next, download the source code to **your PC**.
    - Open a terminal in any working folder and run the following command, replacing `YOUR_USERNAME` with your GitHub username.
    ```bash
    git clone https://github.com/YOUR_USERNAME/virtual-robot-race.git
    ```
    - This creates a `virtual-robot-race` folder on your PC.

### Step 2: Link to the Upstream Repository
Register the original repository as "upstream" so you can easily pull in future updates.

1.  In the terminal, navigate to the project folder you just created.
    ```bash
    cd virtual-robot-race
    ```
2.  Run the following command to register the original repository as `upstream`.
    ```bash
    git remote add upstream https://github.com/aira-race/virtual-robot-race.git
    ```
3.  Verify the configuration.
    ```bash
    git remote -v
    ```
    If you see both `origin` (your fork) and `upstream` (the original) as shown below, you are done.
    ```
    origin    https://github.com/YOUR_USERNAME/virtual-robot-race.git (fetch)
    origin    https://github.com/YOUR_USERNAME/virtual-robot-race.git (push)
    upstream  https://github.com/aira-race/virtual-robot-race.git (fetch)
    upstream  https://github.com/aira-race/virtual-robot-race.git (push)
    ```

#### How to sync upstream updates to your fork

When the original repository is updated, use the following steps to bring those changes into your fork (origin).

```bash
# 1. Fetch the latest changes from the original (upstream)
git fetch upstream

# 2. Merge the upstream main branch into your local branch
git merge upstream/main

# 3. Push the result to your fork (origin)
git push origin main
```

> **Note**: `git fetch` only downloads information — your local files are not changed yet. `git merge` is what actually applies the changes.

#### How to sync upstream changes while keeping your own algorithm safe

> **Does this sound familiar?**
> "I know the upstream was updated — but what if it overwrites the algorithm I spent so long writing?"

Don't worry. As long as you **save your work to a separate branch before syncing**, nothing will be lost.

**How it works:**
```
main  ──[sync with upstream]──────────────────→ matches latest upstream
                \                            /
dev    ──[your algorithm safely saved here]──[review diff and merge]
```

**Steps:**
```bash
# --- Step A: First, save your current work to a "dev" branch ---
git switch -c dev          # create a branch called "dev" and move to it
git add .
git commit -m "Save my algorithm"
git push origin dev        # back it up to GitHub — now nothing can be lost

# --- Step B: Sync the main branch with upstream ---
git switch main            # go back to main
git fetch upstream         # fetch the latest from the original repo
git merge upstream/main    # bring main up to date with upstream
git push origin main       # update your fork on GitHub

# --- Step C: Go back to dev and merge in the upstream changes ---
git switch dev             # return to dev where your algorithm lives
git merge main             # bring the upstream changes (from main) into dev
```

After running `git merge main`, **VS Code will open a diff editor automatically.**
Your code and the upstream changes appear side by side — just choose which parts to keep.

> **"What if something goes wrong?"**
> As long as your `dev` branch and its GitHub backup exist, you can always get back to where you were.
> Don't be afraid to try.

### Step 3: Set Up the Python Environment
Next, set up the Python development environment for controlling the robot.

1.  Run the setup script `setup_env.bat`.
    ```bash
    .\setup_env.bat
    ```
    > **💡 Why `.\`?**: In PowerShell, typing `setup_env.bat` alone is not recognized as a command. The `.\` prefix means "**the file in the current folder**" — it tells PowerShell to run `setup_env.bat` from the current directory.

    This script automatically creates a virtual environment and installs the required Python libraries. When it completes successfully, a new terminal window with the virtual environment activated will open automatically. If you see `(.venv)` at the beginning of the prompt, it worked.

### Step 4: Download the AI Model
Manually download and place the pre-trained model (`model.pth`) used in AI mode.

1.  [Download `model.pth` here](https://drive.google.com/file/d/1NDL3A2lWDgXdy7OUWctyoR35jtYqthWD/view?usp=sharing).
2.  Copy the downloaded `model.pth` file to **both** of the following folders:
    - `Robot1/models/model.pth`
    - `Robot2/models/model.pth`

### Step 5: Configure VS Code
Finally, configure `Visual Studio Code`.

1.  **Open the project folder**
    - Launch VS Code, go to "File" > "Open Folder", and select the `virtual-robot-race` folder.

2.  **Install recommended extensions**
    - Search for and install the following extensions in the VS Code extensions marketplace:
      - **Python** (by Microsoft): Essential tool for Python development.
      - **Gemini Code Assist** (by Google Cloud): AI-powered coding assistance. You will use this in the latter half of the training. (Sign in with your Google account.)

3.  **Select the Python interpreter**
    - Open the command palette with `Ctrl + Shift + P`, then type `Python: Select Interpreter`.
    - Select the project virtual environment `./.venv/Scripts/python.exe`.

    VS Code will now correctly recognize the project's Python environment.

---

## 3. GPU Acceleration (Optional)
If your PC has an NVIDIA GPU, you can speed up AI training and inference.

1.  Open the virtual environment terminal created by `setup_env.bat` (the one with `(.venv)` at the start of the prompt).
2.  Run the following commands to uninstall the CPU version of PyTorch and install the GPU (CUDA) version.
    ```bash
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```
3.  Check whether the GPU is recognized.
    ```bash
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```
    If you see `CUDA available: True`, you are done.

---

## 4. Final Check
Let's do a test run to confirm everything is set up correctly.

1.  Open a new terminal in VS Code (`Ctrl + Shift + @`). Confirm that `(.venv)` is shown in the prompt.
2.  Run `main.py`.
    ```bash
    python main.py
    ```
3.  Confirm the following:
    - [ ] The Unity simulator **launches automatically**.
    - [ ] Two robots start driving the course automatically (default is rule-based mode).
    - [ ] After the race ends, the `training_data` folder opens and driving data is saved.

4.  Check the saved data. If a folder named `run_YYYYMMDD_HHMMSS` was created inside `Robot1/training_data/`, the data was saved successfully.

Setup is complete! Well done.
Let's move on to the next chapter!
