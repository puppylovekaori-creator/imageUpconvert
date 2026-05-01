@echo off
rem SwinIR model download launcher
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\download_models.ps1"
if errorlevel 1 (
    echo.
    echo Model download failed.
    pause
    exit /b 1
)

echo.
echo Model download completed.
pause
