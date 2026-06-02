@echo off
REM ===== Virtual Robot Race - Environment Setup =====
REM Requires: Python 3.12 or 3.13 (64-bit)
REM   Python 3.14 is NOT supported yet (torch/pandas/numpy/opencv have no wheels for 3.14).
REM   https://www.python.org/downloads/windows/  (use "Stable Releases" section, not "Latest")
REM   IMPORTANT: Check "Add Python to PATH" during installation

REM ===== Step 0: Verify Python version (must be 3.12 or 3.13) =====
echo Detected Python:
python --version
where python
echo.

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if not "%PYMAJOR%"=="3" goto :pyver_bad
if "%PYMINOR%"=="12" goto :pyver_ok
if "%PYMINOR%"=="13" goto :pyver_ok
goto :pyver_bad

:pyver_bad
echo.
echo ERROR: Unsupported Python version (%PYVER%).
echo This project requires Python 3.12 or 3.13.
echo.
echo   - Python 3.14+ is NOT supported: torch/pandas/numpy/opencv do not provide
echo     pre-built wheels for 3.14, so pip install will hang or fail.
echo   - Python 3.11 and earlier are also unsupported.
echo.
echo   Fix:
echo   1. Install Python 3.12 from https://www.python.org/downloads/windows/
echo      (scroll to "Stable Releases" - do NOT use the "Latest Python" link)
echo   2. Uninstall Python 3.14 (or ensure 3.12 comes first in PATH)
echo   3. Open a NEW terminal and verify: python --version
echo   4. Re-run setup_env.bat
echo.
pause
exit /b 1

:pyver_ok
echo Python %PYVER% - OK.
echo.

REM ===== Step 1: Create virtual environment =====
python -m venv .venv
if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment.
    echo   - Was "Add Python to PATH" checked during installation?
    echo   - Windows Store Python is NOT supported. Install from python.org instead.
    echo   - If Anaconda/Miniconda is installed, open "Anaconda Prompt" and run this script there.
    echo.
    pause
    exit /b 1
)

REM ===== Step 2: Upgrade pip =====
call .venv\Scripts\python.exe -m pip install --upgrade pip

REM ===== Step 3: Install requirements =====
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some packages.
    echo   - Network issue? Check your internet connection and try again.
    echo   - torch/torchvision failed? Do NOT run pip install manually.
    echo     Just re-run this script. If it keeps failing, check:
    echo     https://github.com/aira-race/virtual-robot-race/issues
    echo.
    pause
    exit /b 1
)

REM ===== Step 4: Configure VS Code to use .venv automatically =====
if not exist .vscode mkdir .vscode
(
    echo {
    echo     "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    echo     "python.terminal.activateEnvironment": true
    echo }
) > .vscode\settings.json

REM ===== Step 5: Verify required packages were installed =====
.venv\Scripts\python.exe -c "import pandas, numpy, torch, cv2, websockets" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Setup incomplete - one or more required packages failed to install.
    echo   - Check the pip output above for the failing package.
    echo   - Common causes: network interruption, proxy/firewall, low disk space.
    echo   - Just re-run setup_env.bat to retry.
    echo.
    pause
    exit /b 1
)

REM ===== Step 6: Show Python version =====
echo.
echo =============================================
for /f "tokens=*" %%i in ('.venv\Scripts\python.exe --version') do echo %%i
echo Setup complete!
echo.
echo Next steps:
echo   1. Open config.txt and set your NAME
echo   2. Run: python main.py
echo =============================================

start cmd /k ".venv\Scripts\activate"
