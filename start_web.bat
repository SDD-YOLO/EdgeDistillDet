@echo off
setlocal
chcp 65001 >nul
title EdgeDistillDet - Local UI
pushd "%~dp0"

echo Starting EdgeDistillDet local UI...
echo Open the URL printed below in your browser. Press Ctrl+C to stop.
echo.

set "NEED_BUILD=0"
set "NEED_NPM_INSTALL=0"
if not exist "web\static\dist\app.js" set "NEED_BUILD=1"
if not exist "web\static\dist\app.css" set "NEED_BUILD=1"
if not exist "web\static\dist\assets\material-icons*.woff2" set "NEED_BUILD=1"
if not exist "web\static\dist\assets\material-symbols-outlined*.woff2" set "NEED_BUILD=1"
if not exist "web\node_modules\@fontsource\plus-jakarta-sans\latin-400.css" set "NEED_NPM_INSTALL=1"
if not exist "web\node_modules\@fontsource\noto-sans-sc\chinese-simplified-400.css" set "NEED_NPM_INSTALL=1"
if not exist "web\node_modules\@fontsource\material-icons\400.css" set "NEED_NPM_INSTALL=1"
if not exist "web\node_modules\@fontsource\material-symbols-outlined\400.css" set "NEED_NPM_INSTALL=1"

if "%NEED_BUILD%"=="1" (
  echo [INFO] Frontend assets missing or outdated. Auto-building web UI...
)
if "%NEED_NPM_INSTALL%"=="1" (
  echo [INFO] Frontend dependencies missing. Auto-installing npm packages...
)
if "%NEED_BUILD%%NEED_NPM_INSTALL%" NEQ "00" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] npm not found. Please run installer.bat first.
    set "EXIT_CODE=1"
    goto :END
  )

  pushd "web"
  if not exist "node_modules" set "NEED_NPM_INSTALL=1"
  if "%NEED_NPM_INSTALL%"=="1" (
    echo [INFO] Installing frontend dependencies...
    call npm install
    if errorlevel 1 (
      popd
      set "EXIT_CODE=1"
      goto :END
    )
  )

  echo [INFO] Building frontend assets...
  call npm run build
  if errorlevel 1 (
    popd
    set "EXIT_CODE=1"
    goto :END
  )
  popd
  echo [INFO] Frontend build ready.
  echo.
)

python "web\app.py"
set "EXIT_CODE=%ERRORLEVEL%"
:END
echo.
echo Press any key to exit...
pause >nul
popd
exit /b %EXIT_CODE%
