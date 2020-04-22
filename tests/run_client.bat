@echo off
setlocal EnableExtensions
echo %1 %2 %3
set title="meritraining"

:start_python_files
start %title% "cmd /c %1 %2 && python %3 && conda deactivate"
for /F "TOKENS=1,2,*" %%a in ('tasklist /FI "WINDOWTITLE eq %title%"') do set MyPID=%%b
echo Process started as PID: %MyPID%

:check_python_files
call:infinite %1 %2 %3
goto:check_python_files

:infinite
tasklist /FI "PID eq %MyPID%" | findstr /c:PID > nul
rem findstr /c:PID command added above to confirm that tasklist has found the process (errorlevel = 0). If not (errorlevel = 1).
if %errorlevel% EQU 1 (start %title% "cmd /c %1 %2 && python %3 && conda deactivate")
if %errorlevel% EQU 1 echo Process was killed and re-started as PID: %MyPID%
if %errorlevel% EQU 1 for /F "TOKENS=1,2,*" %%a in ('tasklist /FI "WINDOWTITLE eq %title%"') do set MyPID=%%b