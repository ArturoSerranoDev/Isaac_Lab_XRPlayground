@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Shared Isaac Lab venv (sibling of Isaac_Lab_XRPlayground under ISAAC_SIM)
set "PYTHON=%~dp0..\..\IsaacLab\env_isaaclab\Scripts\python.exe"
set "TASK=Template-Xrplayground-Marl-Direct-v0"

if not exist "%PYTHON%" (
    echo [ERROR] Could not find env_isaaclab Python at:
    echo   %PYTHON%
    echo Edit PYTHON= in this .bat if your env path changed.
    pause
    exit /b 1
)

:menu
cls
echo ============================================
echo   XRPlayground launcher
echo ============================================
echo   Python: %PYTHON%
echo   Task:   %TASK%
echo ============================================
echo.
echo   1^) Launch Isaac  ^(random agent^)
echo   2^) Launch Isaac  ^(zero agent^)
echo   3^) List environments
echo   4^) Open project in Cursor
echo   5^) Open project folder in Explorer
echo   Q^) Quit
echo.
set "CHOICE="
set /p CHOICE=Select option: 

if /i "%CHOICE%"=="1" goto random
if /i "%CHOICE%"=="2" goto zero
if /i "%CHOICE%"=="3" goto list
if /i "%CHOICE%"=="4" goto cursor
if /i "%CHOICE%"=="5" goto explorer
if /i "%CHOICE%"=="q" exit /b 0
echo Invalid option.
timeout /t 2 >nul
goto menu

:random
echo.
echo Starting random agent...
"%PYTHON%" scripts\random_agent.py --task=%TASK%
echo.
pause
goto menu

:zero
echo.
echo Starting zero agent...
"%PYTHON%" scripts\zero_agent.py --task=%TASK%
echo.
pause
goto menu

:list
echo.
"%PYTHON%" scripts\list_envs.py
echo.
pause
goto menu

:cursor
where cursor >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cursor CLI not found on PATH.
    echo Open Cursor manually: File -^> Open Folder -^> this folder
    pause
    goto menu
)
start "" cursor "%cd%"
goto menu

:explorer
start "" explorer "%cd%"
goto menu
