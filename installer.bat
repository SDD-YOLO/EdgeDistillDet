@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_VERSION=unknown"
for /f "delims=" %%V in ('python -c "import main; print(main.__version__)" 2^>nul') do set "PROJECT_VERSION=%%V"
echo [INFO] EdgeDistillDet version: %PROJECT_VERSION%

set "PYTHON_EXE="
where python >nul 2>nul
if not errorlevel 1 (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%P"
    goto :FOUND_PYTHON
  )
)
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_EXE=py -3"
if not defined PYTHON_EXE (
  echo [ERROR] Python executable not found in PATH.
  echo Please install Python 3 and ensure "python" or "py -3" is available.
  exit /b 1
)

:FOUND_PYTHON
if "%PYTHON_EXE%"=="py -3" (
  echo [INFO] Using Python executable: py -3
  py -3 --version
  echo [INFO] Upgrading pip, setuptools, and wheel...
  py -3 -m pip install --upgrade pip setuptools wheel
) else (
  echo [INFO] Using Python executable: %PYTHON_EXE%
  "%PYTHON_EXE%" --version
  echo [INFO] Upgrading pip, setuptools, and wheel...
  "%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  set "GPU_HARDWARE=0"
) else (
  set "GPU_HARDWARE=1"
)

:CHECK_EXISTING_TORCH
set "TORCH_CUDA="
set "TORCH_CHECK_FILE=%TEMP%\edgedistilldet_torch_check.txt"
if exist "%TORCH_CHECK_FILE%" del /q "%TORCH_CHECK_FILE%" 2>nul
if "%PYTHON_EXE%"=="py -3" (
  py -3 -c "import importlib; t=importlib.import_module('torch'); v=t.__version__.split('+')[0]; m=int(v.split('.')[0]); assert 2 <= m < 3; print('CUDA=1' if t.cuda.is_available() else 'CUDA=0')" > "%TORCH_CHECK_FILE%" 2>nul
) else (
  "%PYTHON_EXE%" -c "import importlib; t=importlib.import_module('torch'); v=t.__version__.split('+')[0]; m=int(v.split('.')[0]); assert 2 <= m < 3; print('CUDA=1' if t.cuda.is_available() else 'CUDA=0')" > "%TORCH_CHECK_FILE%" 2>nul
)
if exist "%TORCH_CHECK_FILE%" (
  set /p "TORCH_CUDA="<"%TORCH_CHECK_FILE%"
)
if defined TORCH_CUDA (
  echo [INFO] Detected existing PyTorch environment: %TORCH_CUDA%
  if "%TORCH_CUDA%"=="CUDA=1" (
    echo [INFO] Existing CUDA-enabled PyTorch environment is compatible, skipping PyTorch install.
    del /q "%TORCH_CHECK_FILE%" 2>nul
    goto :INSTALL_DEPENDENCIES
  )
  if "%TORCH_CUDA%"=="CUDA=0" (
    echo [INFO] Existing CPU-only PyTorch environment is compatible, skipping PyTorch install.
    del /q "%TORCH_CHECK_FILE%" 2>nul
    goto :INSTALL_DEPENDENCIES
  )
)
echo [INFO] No existing compatible PyTorch environment detected.
del /q "%TORCH_CHECK_FILE%" 2>nul

:INSTALL_MATCHED_TORCH
set "EDGE_TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "EDGE_CUDA_TAG=cpu"
set "CUDA_VER="

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [INFO] nvidia-smi not detected, installing CPU version of PyTorch.
  goto :INSTALL_TORCH
)

for /f "tokens=*" %%L in ('nvidia-smi 2^>nul ^| findstr /i "CUDA Version"') do (
  for /f "tokens=1,2,3,4,5,6,7,8,9 delims= " %%a in ("%%L") do (
    if "%%b"=="Version:" set "CUDA_VER=%%c"
    if "%%d"=="Version:" set "CUDA_VER=%%e"
    if "%%f"=="Version:" set "CUDA_VER=%%g"
    if "%%h"=="Version:" set "CUDA_VER=%%i"
  )
  goto :PARSE_CUDA_DONE
)
:PARSE_CUDA_DONE

if not defined CUDA_VER (
  echo [WARN] Failed to parse CUDA version, installing CPU version of PyTorch.
  goto :INSTALL_TORCH
)

echo [INFO] Detected CUDA version: %CUDA_VER%

for /f "tokens=1,2 delims=." %%A in ("%CUDA_VER%") do (
  set "CUDA_MAJOR=%%A"
  set "CUDA_MINOR=%%B"
)

echo [INFO] CUDA major version: %CUDA_MAJOR%  minor version: %CUDA_MINOR%

if %CUDA_MAJOR% GEQ 13 (
  set "EDGE_TORCH_INDEX=https://download.pytorch.org/whl/cu124"
  set "EDGE_CUDA_TAG=cu124"
) else if %CUDA_MAJOR% GEQ 12 (
  if %CUDA_MINOR% GEQ 4 (
    set "EDGE_TORCH_INDEX=https://download.pytorch.org/whl/cu124"
    set "EDGE_CUDA_TAG=cu124"
  ) else (
    set "EDGE_TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    set "EDGE_CUDA_TAG=cu121"
  )
) else if %CUDA_MAJOR% EQU 11 (
  set "EDGE_TORCH_INDEX=https://download.pytorch.org/whl/cu118"
  set "EDGE_CUDA_TAG=cu118"
) else (
  echo [WARN] CUDA version too old, installing CPU version of PyTorch.
)

:INSTALL_TORCH
set "TORCH_INDEX=%EDGE_TORCH_INDEX%"
set "TORCH_TAG=%EDGE_CUDA_TAG%"
echo [INFO] Installing PyTorch channel: %TORCH_TAG%
if "%PYTHON_EXE%"=="py -3" (
  py -3 -m pip install --force-reinstall torch torchvision torchaudio --index-url %TORCH_INDEX%
) else (
  "%PYTHON_EXE%" -m pip install --force-reinstall torch torchvision torchaudio --index-url %TORCH_INDEX%
)
if errorlevel 1 (
  echo [WARN] Installation failed, falling back to CPU...
  if "%PYTHON_EXE%"=="py -3" (
    py -3 -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ) else (
    "%PYTHON_EXE%" -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  )
  if errorlevel 1 exit /b 1
)
if "%PYTHON_EXE%"=="py -3" (
  py -3 -c "import torch; print('[OK] torch=' + torch.__version__ + ' CUDA=' + str(torch.cuda.is_available()))"
) else (
  "%PYTHON_EXE%" -c "import torch; print('[OK] torch=' + torch.__version__ + ' CUDA=' + str(torch.cuda.is_available()))"
)
if errorlevel 1 (
  echo [ERROR] PyTorch validation failed.
  exit /b 1
)

:INSTALL_DEPENDENCIES
echo [INFO] Installing other Python dependencies from requirements.txt...
if "%PYTHON_EXE%"=="py -3" (
  py -3 -m pip install -r requirements.txt
) else (
  "%PYTHON_EXE%" -m pip install -r requirements.txt
)
if errorlevel 1 (
  echo [ERROR] Failed to install Python dependencies from requirements.txt.
  exit /b 1
)

echo [INFO] Installer completed successfully.
exit /b 0
