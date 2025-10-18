@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >NUL 2>&1
title Stable Diffusion WebUI - Codex Installer (Windows)

REM Load optional user settings
if exist webui.settings.bat (
    call webui.settings.bat
)

if not defined PYTHON (set PYTHON=python)

echo.
echo [Codex] Stable Diffusion WebUI installer
echo ----------------------------------------

REM 1) Python availability
echo [1/6] Checking Python...
mkdir tmp 2>NUL
%PYTHON% -c "" >tmp\stdout.txt 2>tmp\stderr.txt
if %ERRORLEVEL% NEQ 0 (
  echo Couldn't launch python
  goto :show_stdout_stderr
)

REM 2) Python version (recommended 3.12)
set "_TMPVER=%TEMP%\codex_pyver_%RANDOM%.txt"
%PYTHON% -c "import sys;print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))" > "%_TMPVER%" 2>NUL
set /p PYVER=<"%_TMPVER%"
del /q "%_TMPVER%" >NUL 2>&1
if "%PYVER%"=="3.12" (
  echo [OK] Python %PYVER%
) else (
  echo [WARN] Detected Python %PYVER%. Recommended: Python 3.12.x
)

REM 3) Create/activate venv
if not defined VENV_DIR set "VENV_DIR=%~dp0venv"
dir "%VENV_DIR%\Scripts\Python.exe" >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
  for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
  echo [2/6] Creating venv in %VENV_DIR% using %PYTHON_FULLNAME%
  %PYTHON_FULLNAME% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Unable to create venv at %VENV_DIR%
    exit /b 1
  )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate venv at %VENV_DIR%
  exit /b 1
)
set PYTHON="%VENV_DIR%\Scripts\Python.exe"

REM 4) Upgrade pip/setuptools/wheel once per minor version
set "PIP_STAMP=%VENV_DIR%\codex_pip_ok.%PYVER%"
if exist "%PIP_STAMP%" (
  echo [3/6] pip/setuptools/wheel up-to-date; skipping.
) else (
  echo [3/6] Upgrading pip/setuptools/wheel ...
  %PYTHON% -m pip install --upgrade pip setuptools wheel
  if errorlevel 1 (
    echo [WARN] Failed to upgrade pip/setuptools/wheel. Continuing.
  ) else (
    echo.>"%PIP_STAMP%"
  )
)

REM 5) Install pinned Torch (CUDA 12.8) BEFORE requirements
if not defined TORCH_INDEX_URL set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
set "TORCH_SPEC=torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"
set "TORCH_STAMP=%VENV_DIR%\codex_torch_cu128.ok"
set "TORCH_MARK=torch==2.7.1|torchvision==0.22.1|torchaudio==2.7.1|index=%TORCH_INDEX_URL%"
set "TORCH_PREV="
if exist "%TORCH_STAMP%" set /p TORCH_PREV=<"%TORCH_STAMP%"
if /I not "%TORCH_PREV%"=="%TORCH_MARK%" (
  echo [4/6] Installing %TORCH_SPEC% from %TORCH_INDEX_URL% ...
  %PYTHON% -m pip install --upgrade --force-reinstall --index-url %TORCH_INDEX_URL% %TORCH_SPEC%
  if errorlevel 1 (
    echo [ERROR] Failed to install Torch CUDA 12.8 wheels.
    exit /b 1
  )
  echo %TORCH_MARK%>"%TORCH_STAMP%"
) else (
  echo [4/6] Torch CUDA 12.8 already pinned; skipping.
)

REM 6) Install project requirements if file present
if exist requirements_versions.txt (
  call :sha256 requirements_versions.txt REQ_HASH
  set "REQ_STAMP=%VENV_DIR%\codex_req.sha256"
  set "REQ_PREV="
  if exist "%REQ_STAMP%" set /p REQ_PREV=<"%REQ_STAMP%"
  if /I "%REQ_PREV%"=="%REQ_HASH%" (
    echo [5/6] Requirements unchanged; skipping install.
  ) else (
    echo [5/6] Installing requirements from requirements_versions.txt ...
    %PYTHON% -m pip install -r requirements_versions.txt
    if errorlevel 1 (
      echo [WARN] Failed to install some requirements. Verification will catch issues.
    ) else (
      echo %REQ_HASH%>"%REQ_STAMP%"
    )
  )
) else (
  echo [5/6] No requirements_versions.txt; skipping.
)

REM Verify core libraries
echo [6/6] Verifying core libraries ...
set "TMPPY=%TEMP%\codex_verify_%RANDOM%.py"
>"%TMPPY%" echo import importlib,sys
>>"%TMPPY%" echo ok=True
>>"%TMPPY%" echo mods='torch torchvision diffusers gradio fastapi huggingface_hub numpy pydantic'.split()
>>"%TMPPY%" echo print('[Verify] Core libraries:')
>>"%TMPPY%" echo for m in mods:
>>"%TMPPY%" echo ^    try:
>>"%TMPPY%" echo ^        mod=importlib.import_module(m);print('[OK]',m,getattr(mod,'__version__','unknown'))
>>"%TMPPY%" echo ^    except Exception as e:
>>"%TMPPY%" echo ^        ok=False;print('[MISS]',m,e)
>>"%TMPPY%" echo sys.exit(0 if ok else 1)

%PYTHON% "%TMPPY%"
set ERR=%ERRORLEVEL%
del /q "%TMPPY%" >NUL 2>&1
if not %ERR%==0 (
  echo [ERROR] One or more core libraries are missing or broken.
  exit /b 1
)

echo.
echo [DONE] Installation and verification completed successfully.
echo       You can now run: run-webui.bat
exit /b 0

:show_stdout_stderr
echo.
echo exit code: %errorlevel%
for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :end
echo.
echo stderr:
type tmp\stderr.txt

:end
exit /b %errorlevel%

:sha256
REM :: Computes SHA256 or falls back to size+mtime
setlocal EnableDelayedExpansion
set "_FILE=%~1"
set "_OUTVAR=%~2"
set "_HASH="
for /f "skip=1 tokens=1 delims= " %%H in ('certutil -hashfile "%_FILE%" SHA256 ^| findstr /R "^[0-9A-F]"') do if not defined _HASH set "_HASH=%%H"
if not defined _HASH (
  for %%A in ("%_FILE%") do set "_HASH=%%~zA_%%~tA"
)
endlocal & set "%_OUTVAR%=%_HASH%"
exit /b 0

