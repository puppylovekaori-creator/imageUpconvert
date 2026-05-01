@echo off
setlocal
cd /d "%~dp0"

call :find_python
if not defined PYTHON_CMD (
    echo Python was not found, or only an unsupported version was found.
    echo Please install Python 3.10 to 3.13, then run setup.bat again.
    pause
    exit /b 1
)

echo Using %PYTHON_CMD%

set "VENV_DIR=%LOCALAPPDATA%\imageUpconvert\venv"
echo Virtual environment: %VENV_DIR%

if not exist "%VENV_DIR%\Scripts\python.exe" (
    if not exist "%LOCALAPPDATA%\imageUpconvert" mkdir "%LOCALAPPDATA%\imageUpconvert"
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create venv.
        pause
        exit /b 1
    )
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

for /d %%D in ("%VENV_DIR%\Lib\site-packages\~ip*") do (
    rd /s /q "%%~fD"
)

"%VENV_PYTHON%" -m ensurepip --upgrade
if errorlevel 1 goto :pip_error

"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :pip_error

"%VENV_PYTHON%" -m pip show torch >nul 2>nul
if errorlevel 1 (
    echo Installing CPU PyTorch as the default first setup...
    echo If you want CUDA later, replace torch using the command from the official PyTorch site.
    "%VENV_PYTHON%" -m pip install --disable-pip-version-check torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto :pip_error
) else (
    echo torch is already installed. Skipping CPU reinstall.
)

echo Installing timm after torch so pip does not choose a PyTorch build automatically...
"%VENV_PYTHON%" -m pip install --disable-pip-version-check timm==0.9.16 --no-deps
if errorlevel 1 goto :pip_error

echo.
echo Downloading official SwinIR models for the GUI-supported 2x and 4x modes...
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\tools\download_models.ps1"
if errorlevel 1 goto :pip_error

echo.
echo Environment check:
"%VENV_PYTHON%" -m app.env_check

echo.
echo Setup completed.
echo Next steps:
echo 1. Double-click run_gui.bat
echo.
echo venv location:
echo %VENV_DIR%
pause
exit /b 0

:find_python
set "PYTHON_CMD="
for %%V in (3.13 3.12 3.11 3.10) do (
    py -%%V -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -%%V"
        goto :eof
    )
)
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) and sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
)
goto :eof

:pip_error
echo Installation failed.
echo Review the error above. If this is a PyTorch GPU setup case, use the command from:
echo https://pytorch.org/get-started/locally/
pause
exit /b 1
