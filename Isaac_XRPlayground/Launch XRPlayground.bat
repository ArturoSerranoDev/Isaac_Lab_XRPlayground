@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Shared Isaac Lab venv (sibling of Isaac_Lab_XRPlayground under ISAAC_SIM)
set "PYTHON=%~dp0..\..\IsaacLab\env_isaaclab\Scripts\python.exe"
set "MARL_TASK=Template-Xrplayground-Marl-Direct-v0"
set "CART_TASK=Template-Xrplayground-Cartpole-Direct-v0"

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
echo ============================================
echo.
echo   -- Cartpole (single-agent) --
echo   1^) Train        ^(headless, fast^)
echo   2^) Play         ^(watch trained policy^)
echo   3^) Random agent ^(just watch physics^)
echo.
echo   -- MARL cart+pendulum (template) --
echo   4^) Random agent
echo   5^) Zero agent
echo.
echo   -- Utils --
echo   6^) List environments
echo   7^) Open project in Cursor
echo   8^) Open project folder in Explorer
echo   Q^) Quit
echo.
set "CHOICE="
set /p CHOICE=Select option: 

if /i "%CHOICE%"=="1" goto cart_train
if /i "%CHOICE%"=="2" goto cart_play
if /i "%CHOICE%"=="3" goto cart_random
if /i "%CHOICE%"=="4" goto marl_random
if /i "%CHOICE%"=="5" goto marl_zero
if /i "%CHOICE%"=="6" goto list
if /i "%CHOICE%"=="7" goto cursor
if /i "%CHOICE%"=="8" goto explorer
if /i "%CHOICE%"=="q" exit /b 0
echo Invalid option.
timeout /t 2 >nul
goto menu

:cart_train
echo.
echo Training cartpole (headless)...
"%PYTHON%" scripts\skrl\train.py --task=%CART_TASK% --headless
echo.
pause
goto menu

:cart_play
echo.
echo Playing latest cartpole checkpoint...
"%PYTHON%" scripts\skrl\play.py --task=%CART_TASK% --num_envs=16
echo.
pause
goto menu

:cart_random
echo.
echo Cartpole random agent...
"%PYTHON%" scripts\random_agent.py --task=%CART_TASK% --num_envs=16
echo.
pause
goto menu

:marl_random
echo.
echo MARL random agent...
"%PYTHON%" scripts\random_agent.py --task=%MARL_TASK% --num_envs=16
echo.
pause
goto menu

:marl_zero
echo.
echo MARL zero agent...
"%PYTHON%" scripts\zero_agent.py --task=%MARL_TASK% --num_envs=16
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
