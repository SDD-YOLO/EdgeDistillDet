@echo off
setlocal
chcp 65001 >nul
set "PROJECT_VERSION=unknown"
for /f "delims=" %%V in ('python -c "import main; print(main.__version__)" 2^>nul') do set "PROJECT_VERSION=%%V"
title EdgeDistillDet - Local UI v%PROJECT_VERSION%
pushd "%~dp0"

echo Starting EdgeDistillDet local UI...
echo EdgeDistillDet version: %PROJECT_VERSION%
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

set "PYTHON_CMD=python"
set "PYTHON_EXE="
where %PYTHON_CMD% >nul 2>nul
if not errorlevel 1 (
  for /f "delims=" %%P in ('where %PYTHON_CMD% 2^>nul') do (
    set "PYTHON_EXE=%%P"
    goto :FOUND_PYTHON
  )
)
if not defined PYTHON_EXE (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=py -3"
    goto :FOUND_PYTHON
  )
)
:FOUND_PYTHON
if not defined PYTHON_EXE (
  echo [ERROR] Python executable not found in PATH.
  echo Please install Python 3 and ensure "python" or "py" is available in PATH.
  set "EXIT_CODE=1"
  goto :END
)
echo [INFO] Using Python executable: %PYTHON_EXE%
set "PYTHON_OK=1"
if "%PYTHON_EXE%"=="py -3" (
  py -3 -m pip show fastapi >nul 2>nul || set "PYTHON_OK=0"
  if "%PYTHON_OK%"=="1" py -3 -m pip show uvicorn >nul 2>nul || set "PYTHON_OK=0"
) else (
  "%PYTHON_EXE%" -m pip show fastapi >nul 2>nul || set "PYTHON_OK=0"
  if "%PYTHON_OK%"=="1" "%PYTHON_EXE%" -m pip show uvicorn >nul 2>nul || set "PYTHON_OK=0"
)
if "%PYTHON_OK%"=="0" (
  if not "%PYTHON_EXE%"=="py -3" (
    where py >nul 2>nul
    if not errorlevel 1 (
      py -3 -m pip show fastapi >nul 2>nul && py -3 -m pip show uvicorn >nul 2>nul
      if not errorlevel 1 (
        set "PYTHON_EXE=py -3"
        echo [INFO] FastAPI and Uvicorn found with %PYTHON_EXE%, switching interpreter.
        goto :RUN_PYTHON
      )
    )
  )
  echo [ERROR] FastAPI and/or Uvicorn are not installed for %PYTHON_EXE%.
  echo Install them with: "%PYTHON_EXE%" -m pip install -r requirements.txt
  set "EXIT_CODE=1"
  goto :END
)
:RUN_PYTHON
if "%PYTHON_EXE%"=="py -3" (
  py -3 "web\app.py"
) else (
  "%PYTHON_EXE%" "web\app.py"
)
set "EXIT_CODE=%ERRORLEVEL%"
:END
echo.
echo Press any key to exit...
pause >nul
popd
exit /b %EXIT_CODE%
