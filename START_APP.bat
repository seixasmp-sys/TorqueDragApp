@echo off
title Torque and Drag Sensitivity App
cd /d "%~dp0"

echo Starting Torque and Drag Sensitivity App...
echo.
echo If this is your first time running the app, dependencies will be checked/installed.
echo.

python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Streamlit was not found. Installing requirements...
    python -m pip install -r requirements.txt
)

echo.
echo Opening the application at http://localhost:8501
start http://localhost:8501

python -m streamlit run app.py --server.headless true --server.port 8501

pause
