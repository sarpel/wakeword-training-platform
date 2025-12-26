@echo off
echo Starting Wakeword Training Platform...

:: Activate virtual environment
call venv\Scripts\activate

:: Set PYTHONPATH to current directory to ensure imports work
set PYTHONPATH=%PYTHONPATH%;%CD%

:: Run the application
python src/ui/app.py

pause
