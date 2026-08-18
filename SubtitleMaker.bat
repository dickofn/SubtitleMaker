@echo off
rem Launch SubtitleMaker using its own virtual environment.
rem pythonw.exe runs the Tk GUI without leaving a console window behind it.

set "VENV_PY=%~dp0.venv\Scripts\pythonw.exe"

if not exist "%VENV_PY%" (
    echo Virtual environment not found at:
    echo   %VENV_PY%
    echo.
    echo Recreate it with:
    echo   py -3.13 -m venv "%~dp0.venv"
    echo   "%~dp0.venv\Scripts\python.exe" -m pip install faster-whisper srt
    echo.
    pause
    exit /b 1
)

start "" "%VENV_PY%" "%~dp0SubtitleMaker.py"
