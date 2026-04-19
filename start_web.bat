@echo off
chcp 65001 >nul
title EdgeDistillDet - Local UI
cd /d "%~dp0"

echo 正在启动 EdgeDistillDet 本地界面...
echo 浏览器可访问控制台中打印的地址；停止请关闭本窗口或按 Ctrl+C。
echo.

python web\app.py
echo.
pause
