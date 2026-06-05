@echo off
title Torque and Drag Sensitivity App - Port 8502
cd /d "%~dp0"

echo Starting Torque and Drag Sensitivity App on port 8502...
echo.

python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Streamlit was not found. Installing requirements...
    python -m pip install -r requirements.txt
)

start http://localhost:8502
python -m streamlit run app.py --server.headless true --server.port 8502

pause
