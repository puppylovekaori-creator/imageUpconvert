@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\download_models.ps1"
if errorlevel 1 (
    echo.
    echo AI Upscale backend download failed.
    pause
    exit /b 1
)

echo.
echo AI Upscale backend download completed.
pause
