"""
启动脚本 - 确保从正确路径加载app.py
修复 Windows 路径别名问题 (Personal_Files -> Personal_F_Files)
"""
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

# 强制使用绝对路径加载
APP_PATH = Path(r'D:\Personal_Files\Projects\EdgeDistillDet\web\app.py')
print(f"[BOOT] Loading Flask app from: {APP_PATH}")
print(f"[BOOT] File exists: {APP_PATH.exists()}")

if not APP_PATH.exists():
    print(f"[ERROR] File not found: {APP_PATH}")
    sys.exit(1)

# 使用 SourceFileLoader 加载，绕过所有缓存
loader = SourceFileLoader("app", str(APP_PATH))
app_module = loader.load_module()

print(f"[BOOT] Starting server...")
app_module.app.run(host='0.0.0.0', port=5000, debug=True)
