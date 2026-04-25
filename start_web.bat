@echo off
setlocal
chcp 65001 >nul
title EdgeDistillDet - Local UI
pushd "%~dp0"

echo Starting EdgeDistillDet local UI...
echo Open the URL printed below in your browser. Press Ctrl+C to stop.
echo.

python "web\app.py"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Press any key to exit...
pause >nul
popd
exit /b %EXIT_CODE%
