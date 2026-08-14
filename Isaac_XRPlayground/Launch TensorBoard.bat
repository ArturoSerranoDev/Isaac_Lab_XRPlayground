@echo off
setlocal EnableExtensions
title Isaac Lab XRPlayground - TensorBoard
cd /d "%~dp0"

set "PYTHON=%~dp0..\..\IsaacLab\env_isaaclab\Scripts\python.exe"
rem RSL-RL is the current trainer. SKRL runs (if any) live under logs\skrl.
set "LOGDIR=%~dp0logs\rsl_rl"
set "PORT=6006"

if not exist "%PYTHON%" (
    echo.
    echo  [ERROR] env_isaaclab Python not found:
    echo    %PYTHON%
    echo.
    echo  Edit PYTHON in this .bat if your Isaac Lab env lives elsewhere.
    pause
    exit /b 1
)

if not exist "%LOGDIR%" (
    echo.
    echo  [WARN] No RSL-RL logs yet:
    echo    %LOGDIR%
    echo  Train once first; TensorBoard will open an empty dashboard until then.
    echo.
)

echo.
echo  Freeing port %PORT% if an old TensorBoard is still running...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    echo  Stopping PID %%P
    taskkill /PID %%P /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo.
echo  Starting TensorBoard...
echo  Logs: %LOGDIR%
echo  Open: http://localhost:%PORT%
echo  Press Ctrl+C to stop.
echo.

start "" "http://localhost:%PORT%"
"%PYTHON%" -m tensorboard.main --logdir="%LOGDIR%" --port=%PORT% --reload_interval=15
if errorlevel 1 (
    echo.
    echo  [ERROR] TensorBoard failed. Is tensorboard installed in env_isaaclab?
    echo    "%PYTHON%" -m pip install tensorboard
    pause
)
exit /b %ERRORLEVEL%
