@echo off
cd /d "%~dp0"
echo Installing/updating required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Error installing dependencies. Try running this file again as Administrator.
  pause
  exit /b 1
)
echo.
echo Starting T&D Complete App...
python -m streamlit run app.py
pause
