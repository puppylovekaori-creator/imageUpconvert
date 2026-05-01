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
echo Virtual environment: LOCALAPPDATA\imageUpconvert\venv

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

for %%P in ("%VENV_DIR%\Lib\site-packages\~ip*" "%VENV_DIR%\Lib\site-packages\~il*") do (
    for /d %%D in ("%%~fP") do (
    rd /s /q "%%~fD"
    )
)

"%VENV_PYTHON%" -m ensurepip --upgrade
if errorlevel 1 goto :pip_error

"%VENV_PYTHON%" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto :pip_error

"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :pip_error

echo.
echo Downloading GIMP AI Upscale compatible backend and models...
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\tools\download_models.ps1"
if errorlevel 1 goto :pip_error

echo.
echo GUI smoke test:
"%VENV_PYTHON%" -m app.main --smoke-test
if errorlevel 1 goto :pip_error

echo.
echo Setup completed.
echo Next steps:
echo 1. Double-click run_gui.bat
echo 2. Set GIMP path, input folder, and output folder in the GUI.
echo 3. Run comparison first, then test_5, then test_20, then batch.
echo.
echo Venv location: LOCALAPPDATA\imageUpconvert\venv
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
echo Review the error above.
pause
exit /b 1
