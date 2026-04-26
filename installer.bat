@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title EdgeDistillDet - New User Setup
cd /d "%~dp0"

REM ============================================================
REM  Helper: GET_TIME_CS
REM ============================================================
goto :MAIN

:PRINT_PROGRESS
set /a "_pct=STEPS_DONE*100/TOTAL_STEPS"
set "_bar="
set /a "_filled=STEPS_DONE*20/TOTAL_STEPS"
for /l %%i in (1,1,20) do (
    if %%i leq !_filled! (
        set "_bar=!_bar!#"
    ) else (
        set "_bar=!_bar!-"
    )
)
echo   [!_bar!] !_pct!%% (!STEPS_DONE!/!TOTAL_STEPS!)
exit /b 0

:MARK_STEP_DONE
set /a STEPS_DONE+=1
exit /b 0

:PRINT_BANNER
python -c "from main import BANNER; print(BANNER)"
exit /b 0

REM ============================================================
REM  MAIN
REM ============================================================
:MAIN

echo.
echo ==============================================
echo  EdgeDistillDet Installer
echo ==============================================
echo.

REM -- 1. Detect Python --
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Please install Python 3.10+ and ensure it is in PATH.
    echo         Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM -- 2. Verify Python version >= 3.10 --
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "PY_FULL=%%V"
for /f "tokens=1,2 delims=." %%A in ("%PY_FULL%") do (
    set /a "PY_MAJOR=%%A"
    set /a "PY_MINOR=%%B"
)
if !PY_MAJOR! lss 3 goto :PY_VER_FAIL
if !PY_MAJOR! equ 3 if !PY_MINOR! lss 10 goto :PY_VER_FAIL
goto :PY_VER_OK

:PY_VER_FAIL
echo [ERROR] Python %PY_FULL% detected, but 3.10+ is required.
echo         Download: https://www.python.org/downloads/
pause
exit /b 1

:PY_VER_OK
echo   Python %PY_FULL% detected. OK
echo.

REM -- 3. Detect pip --
python -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pip is unavailable in the current Python installation.
    pause
    exit /b 1
)

REM -- Init progress --
set /a TOTAL_STEPS=2
set /a STEPS_DONE=0

REM ============================================
REM  Print Banner (after deps are ready)
REM ============================================
call :PRINT_BANNER
echo.

REM ============================================
REM  [1/2] Upgrade pip
REM ============================================
echo [1/2] Upgrading pip ...
echo.
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)
call :MARK_STEP_DONE
call :PRINT_PROGRESS
echo.

REM ============================================
REM  [2/2] Install Python dependencies
REM ============================================
echo [2/2] Installing Python dependencies from requirements.txt ...
echo.

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in: %CD%
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
call :MARK_STEP_DONE
call :PRINT_PROGRESS
echo.

pause
exit /b 0
