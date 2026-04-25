@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title EdgeDistillDet - New User Setup
cd /d "%~dp0"

echo ==============================================
echo EdgeDistillDet Installer
echo ==============================================
echo.

REM 1) Check Python
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Please install Python 3.10+ and retry.
  goto :FAILED
)

call :PRINT_BANNER

python --version
echo.

REM 2) Check pip
python -m pip --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] pip is unavailable in current Python installation.
  goto :FAILED
)

set /a TOTAL_STEPS=5
set /a STEPS_DONE=0
call :GET_TIME_CS INSTALL_START_CS

echo [1/5] Upgrading pip ...
call :PRINT_PROGRESS
python -m pip install --upgrade pip
if errorlevel 1 goto :FAILED
call :MARK_STEP_DONE
echo.

echo [2/5] Installing Python dependencies from requirements.txt ...
call :PRINT_PROGRESS
python -m pip install -r requirements.txt
if errorlevel 1 goto :FAILED
call :MARK_STEP_DONE
echo.

REM 3) Check npm / Node.js
where npm >nul 2>nul
if errorlevel 1 (
  echo [INFO] npm not found. Trying to install Node.js LTS via winget...
  where winget >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] winget not found. Install Node.js LTS manually:
    echo         https://nodejs.org/
    goto :FAILED
  )

  winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo [ERROR] winget installation failed. Please install Node.js manually.
    goto :FAILED
  )

  echo [INFO] Node.js installed. Refreshing current shell environment...
  call refreshenv >nul 2>nul
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm still not found. Reopen terminal or install Node.js LTS manually.
  goto :FAILED
)

echo [3/5] Node / npm version:
call :PRINT_PROGRESS
node --version
npm --version
if errorlevel 1 goto :FAILED
call :MARK_STEP_DONE
echo.

REM 4) Install frontend dependencies and build
if not exist "web\package.json" (
  echo [ERROR] web\package.json not found. Cannot continue frontend setup.
  goto :FAILED
)

echo [4/5] Installing web dependencies (npm install) ...
call :PRINT_PROGRESS
pushd web
call npm install
if errorlevel 1 (
  popd
  goto :FAILED
)
call :MARK_STEP_DONE
echo.

echo [5/5] Building frontend (npm run build) ...
call :PRINT_PROGRESS
call npm run build
if errorlevel 1 (
  popd
  goto :FAILED
)
if not exist "static\dist\app.js" (
  echo [ERROR] Frontend build output missing: web\static\dist\app.js
  popd
  goto :FAILED
)
if not exist "static\dist\app.css" (
  echo [ERROR] Frontend build output missing: web\static\dist\app.css
  popd
  goto :FAILED
)
if not exist "static\dist\assets\material-icons*.woff2" (
  echo [ERROR] Icon font missing: material-icons*.woff2
  popd
  goto :FAILED
)
if not exist "static\dist\assets\material-symbols-outlined*.woff2" (
  echo [ERROR] Icon font missing: material-symbols-outlined*.woff2
  popd
  goto :FAILED
)
call :MARK_STEP_DONE
popd

call :GET_TIME_CS INSTALL_END_CS
set /a INSTALL_TOTAL_CS=INSTALL_END_CS-INSTALL_START_CS
if !INSTALL_TOTAL_CS! lss 0 set /a INSTALL_TOTAL_CS+=8640000
call :FORMAT_DURATION !INSTALL_TOTAL_CS! INSTALL_TOTAL_STR

echo.
echo ==============================================
echo Setup completed successfully.
echo Frontend build output: web\static\dist
echo Total elapsed time: !INSTALL_TOTAL_STR!
echo ==============================================
echo.
echo Press any key to exit...
pause >nul
exit /b 0

:FAILED
echo.
echo ==============================================
echo Setup failed. Please fix errors above and retry.
echo ==============================================
echo.
echo Press any key to exit...
pause >nul
exit /b 1

:PRINT_BANNER
python -c "print(''' _____    _           ______ _     _   _ _ _______     _   \n|  ___|  | |          |  _  (_)   | | (_) | |  _  \\   | |  \n| |__  __| | __ _  ___| | | |_ ___| |_ _| | | | | |___| |_ \n|  __|/ _` |/ _` |/ _ \\ | | | / __| __| | | | | | / _ \\ __|\n| |__| (_| | (_| |  __/ |/ /| \\__ \\ |_| | | | |/ /  __/ |_ \n\\____/\\__,_|\\__, |\\___|___/ |_|___/\\__|_|_|_|___/ \\___|\\___|\n             __/ |                                          \n            |___/                                           \n\n  \u9762\u5411\u8fb9\u7f18\u8ba1\u7b97\u7684\u5fae\u5c0f\u76ee\u6807\u81ea\u9002\u5e94\u84b8\u998f\u4e0e\u68c0\u6d4b\u8bc4\u4f30\u7cfb\u7edf  v{__version__}\n  Edge-Oriented Micro Small-Target Adaptive Distillation ^& Detection Evaluation System\n''')"
exit /b 0
