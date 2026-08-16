@echo off
setlocal

rem Keep Kit's USD loader away from the inaccessible WindowsApps alias directory.
set "PATH=%PATH:C:\Users\Arturo\AppData\Local\Microsoft\WindowsApps;=%"
set "PATH=%PATH:C:\Users\Arturo\AppData\Local\Microsoft\WindowsApps=%"
set "PXR_USD_WINDOWS_DLL_PATH=%PATH%"

cd /d "%~dp0Isaac_XRPlayground"
"C:\PROYECTOS PERSONALES\ISAAC_SIM\IsaacLab\env_isaaclab\Scripts\python.exe" -u scripts\bridge\run_xr_bridge_spot.py --task=Template-Xrplayground-Spot-Loco-Walk-Play-v0 --num_envs=1 --device cpu --disable_fabric --real-time

pause
