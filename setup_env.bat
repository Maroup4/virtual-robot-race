@echo off
REM ===== Virtual Robot Race - Environment Setup =====
REM Requires: Python 3.12 or later (64-bit)
REM   https://www.python.org/downloads/
REM   IMPORTANT: Check "Add Python to PATH" during installation

REM ===== Step 1: Create virtual environment =====
python -m venv .venv
if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment.
    echo   - Is Python 3.12+ installed?
    echo   - Was "Add Python to PATH" checked during installation?
    echo   - Try: python --version
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
    echo ERROR: Failed to install some packages. Check your internet connection.
    pause
    exit /b 1
)

REM ===== Step 4: Show Python version =====
echo.
echo =============================================
for /f "tokens=*" %%i in ('.venv\Scripts\python.exe --version') do echo %%i
echo Virtual environment setup complete!
echo To activate, run:
echo   .venv\Scripts\activate
echo =============================================

start cmd /k ".venv\Scripts\activate"
