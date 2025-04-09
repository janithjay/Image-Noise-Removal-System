@echo off
echo Image Noise Removal System
echo ========================

REM Check if virtual environment exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
    echo Installing dependencies...
    python install_dependencies.py
) else (
    call .venv\Scripts\activate
)

REM Run the application
echo Starting the application...
python app.py

pause 