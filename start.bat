@echo off
if not exist .venv (
    echo ERROR: Virtual environment not found.
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

REM Sanity check: required packages must be importable.
REM A .venv directory can exist while pip install failed midway (e.g. wrong Python version),
REM which would otherwise surface as a cryptic "No module named pandas" at runtime.
.venv\Scripts\python.exe -c "import pandas, numpy, torch, cv2, websockets" 2>nul
if errorlevel 1 (
    echo ERROR: Required Python packages are missing from .venv.
    echo This usually means setup_env.bat did not finish successfully.
    echo Please re-run setup_env.bat and watch for errors.
    pause
    exit /b 1
)

call .venv\Scripts\activate
python main.py
