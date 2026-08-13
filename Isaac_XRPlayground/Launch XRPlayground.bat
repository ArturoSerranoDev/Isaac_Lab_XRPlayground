@echo off
setlocal EnableExtensions
title Isaac Lab XRPlayground - Launcher
cd /d "%~dp0"

set "PYTHON=%~dp0..\..\IsaacLab\env_isaaclab\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo.
    echo  [ERROR] env_isaaclab not found:
    echo    %PYTHON%
    echo.
    echo  Activate your Isaac Lab environment or edit this .bat.
    pause
    exit /b 1
)

"%PYTHON%" scripts\xr_launcher.py
exit /b %ERRORLEVEL%
