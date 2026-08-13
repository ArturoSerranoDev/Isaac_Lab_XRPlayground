@echo off
setlocal EnableExtensions
title Isaac Lab XRPlayground - TensorBoard
cd /d "%~dp0"

set "PYTHON=%~dp0..\..\IsaacLab\env_isaaclab\Scripts\python.exe"
set "LOGDIR=%~dp0logs\skrl"

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
    echo  [WARN] No skrl logs yet:
    echo    %LOGDIR%
    echo  Train once first; TensorBoard will open an empty dashboard until then.
    echo.
)

echo.
echo  Starting TensorBoard...
echo  Logs: %LOGDIR%
echo  Open: http://localhost:6006
echo  Press Ctrl+C to stop.
echo.

start "" "http://localhost:6006"
"%PYTHON%" -m tensorboard.main --logdir="%LOGDIR%" --port=6006
if errorlevel 1 (
    echo.
    echo  [ERROR] TensorBoard failed. Is tensorboard installed in env_isaaclab?
    echo    "%PYTHON%" -m pip install tensorboard
    pause
)
exit /b %ERRORLEVEL%
