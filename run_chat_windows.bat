@echo off
REM Windows convenience launcher for the math theorem correction chat.
REM Requires: Python 3.10+, NVIDIA GPU with CUDA, dependencies installed via:
REM   pip install -r requirements-windows.txt
REM
REM Double-click to start, or run from a terminal.

cd /d "%~dp0"
python scripts\inference\chat_windows.py %*
pause
