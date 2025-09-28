@echo off
REM ===== Step 1: Create virtual environment with Python 3.10 =====
py -3.10 -m venv .venv

REM ===== Step 2: Upgrade pip =====
call .venv\Scripts\python.exe -m pip install --upgrade pip

REM ===== Step 3: Install requirements =====
call .venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo =============================================
echo Virtual environment setup complete! (Python 3.10)
echo To activate, run:
echo   .venv\Scripts\activate
echo =============================================

start cmd /k ".venv\Scripts\activate"
