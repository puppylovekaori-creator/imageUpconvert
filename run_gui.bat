@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%LOCALAPPDATA%\imageUpconvert\venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo venv was not found at:
    echo %VENV_DIR%
    echo Please run setup.bat first.
    pause
    exit /b 1
)

"%VENV_DIR%\Scripts\python.exe" -m app.main
if errorlevel 1 (
    echo.
    echo The GUI exited with an error.
    pause
    exit /b 1
)
