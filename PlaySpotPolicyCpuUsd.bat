@echo off
setlocal

rem Isaac Sim's DLL bootstrap cannot add the WindowsApps alias directory.
set "PATH=%PATH:C:\Users\Arturo\AppData\Local\Microsoft\WindowsApps;=%"
set "PATH=%PATH:C:\Users\Arturo\AppData\Local\Microsoft\WindowsApps=%"
set "PXR_USD_WINDOWS_DLL_PATH=%PATH%"

cd /d "%~dp0Isaac_XRPlayground"
"C:\PROYECTOS PERSONALES\ISAAC_SIM\IsaacLab\env_isaaclab\Scripts\python.exe" -u scripts\rsl_rl\play.py --task=Template-Xrplayground-Spot-Loco-Walk-Play-v0 --checkpoint="%~dp0Isaac_XRPlayground\logs\rsl_rl\xrplayground_spot_loco\2026-08-15_20-54-35\model_2499.pt" --num_envs=1 --device cpu --disable_fabric

pause
