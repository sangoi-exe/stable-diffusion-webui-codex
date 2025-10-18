@echo off
setlocal EnableExtensions
chcp 65001 >NUL 2>&1
title Stable Diffusion WebUI - Codex Launcher
color 0A

REM Align with upstream webui.bat flow and add verify steps
if exist webui.settings.bat (
    call webui.settings.bat
)

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0.venv")

echo.
echo [Codex] Stable Diffusion WebUI launcher (Windows)
echo ---------------------------------------------------

REM 1) Python availability
echo [1/7] Checking Python in PATH...
where %PYTHON% >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found in PATH. Install Python 3.10 (64-bit) and re-run.
  exit /b 1
)

REM 2) Python version (expect 3.10.x)
set "_TMPVER=%TEMP%\codex_pyver_%RANDOM%.txt"
%PYTHON% -c "import sys;print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))" > "%_TMPVER%" 2>NUL
set /p PYVER=<"%_TMPVER%"
del /q "%_TMPVER%" >NUL 2>&1
if not "%PYVER%"=="3.10" (
  echo [ERROR] Detected Python %PYVER%. Windows build requires Python 3.10.x.
  exit /b 1
)
echo [OK] Python %PYVER%

REM 3) VENV create/activate (same style as webui.bat)
if ["%VENV_DIR%"] == ["-"] goto :venv_done
if ["%SKIP_VENV%"] == ["1"] goto :venv_done

dir "%VENV_DIR%\Scripts\Python.exe" >NUL 2>&1
if %ERRORLEVEL% == 0 goto :venv_activate

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo [2/7] Creating venv in %VENV_DIR% using %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%"
if errorlevel 1 (
  echo [ERROR] Unable to create venv at %VENV_DIR%
  exit /b 1
)

:venv_activate
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate venv at %VENV_DIR%
  exit /b 1
)
set PYTHON="%VENV_DIR%\Scripts\Python.exe"

:venv_done

REM 4) Upgrade pip/setuptools/wheel
echo [3/7] Upgrading pip/setuptools/wheel ...
%PYTHON% -m pip install --upgrade pip setuptools wheel >NUL
if errorlevel 1 (
  echo [WARN] Failed to upgrade pip/setuptools/wheel. Continuing.
)

REM 5) Install requirements (best-effort) then verify imports
if exist requirements_versions.txt (
  echo [4/7] Installing requirements from requirements_versions.txt ...
  %PYTHON% -m pip install -r requirements_versions.txt
  if errorlevel 1 (
    echo [WARN] Failed to install some requirements. Will verify imports next.
  )
)

echo [5/7] Verifying core libraries ...
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
  echo [ERROR] One or more core libraries are missing. Please run:
  echo         %PYTHON% -m pip install -r requirements_versions.txt
  exit /b 1
)

REM 6) Optional tools
echo [6/7] Checking optional tools (ffmpeg) ...
where ffmpeg >NUL 2>&1
if errorlevel 1 (
  echo [WARN] ffmpeg not found (video export disabled)
) else (
  echo [OK] ffmpeg found (video export ready)
)

REM 7) Launch (reusing upstream webui.bat behavior)
echo [7/7] Starting WebUI ...
%PYTHON% launch.py %*
if exist tmp\restart goto :venv_done
exit /b %ERRORLEVEL%
