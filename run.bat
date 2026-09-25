@echo off
rem Double-click launcher for the Hand Tracker.
rem Any arguments are passed straight through, e.g.:
rem     run.bat --camera 1
rem     run.bat --image tests\fixtures\hand.png

setlocal

cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo.
    echo   The virtual environment is missing.
    echo.
    echo   Create it with:
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PYTHON%" hand_tracker.py %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo   Hand Tracker exited with code %EXITCODE%.
    pause
)

exit /b %EXITCODE%
