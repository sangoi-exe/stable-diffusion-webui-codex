@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >NUL 2>&1
title Stable Diffusion WebUI — Codex Launcher
color 0A

REM Repo root (folder of this script)
set "ROOT=%~dp0"
pushd "%ROOT%"

echo.
echo [Codex] Stable Diffusion WebUI launcher (Windows)
echo ---------------------------------------------------

REM 1) Check Python in PATH
echo [1/7] Checking Python in PATH...
where python >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found in PATH. Install Python 3.10 (64-bit) and re-run.
  echo         https://www.python.org/downloads/release/python-3106/
  exit /b 1
)

REM 2) Check Python version (Windows builds target 3.10)
for /f %%v in ('python -c "import sys;print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))"') do set PYVER=%%v
if not "%PYVER%"=="3.10" (
  echo [ERROR] Detected Python %PYVER%. Windows build requires Python 3.10.x.
  echo         Please install Python 3.10 (64-bit) and ensure it is first in PATH.
  exit /b 1
)
echo [OK] Python %PYVER%

REM 3) Create venv if missing and activate
if not exist ".venv\Scripts\python.exe" (
  echo [2/7] Creating virtual environment at .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv at .venv
    exit /b 1
  )
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate venv .venv
  exit /b 1
)

REM 4) Upgrade pip/setuptools/wheel
echo [3/7] Upgrading pip/setuptools/wheel ...
python -m pip install --upgrade pip setuptools wheel >NUL
if errorlevel 1 (
  echo [ERROR] pip upgrade failed.
  exit /b 1
)

REM 5) Install project requirements
echo [4/7] Installing project requirements ...
if exist requirements_versions.txt (
  python -m pip install -r requirements_versions.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install requirements from requirements_versions.txt
    exit /b 1
  )
) else (
  echo [WARN] requirements_versions.txt not found. Skipping.
)

REM 6) Verify core libraries (torch, torchvision, diffusers, gradio, fastapi, huggingface_hub, numpy, pydantic)
echo [5/7] Verifying core libraries ...
set "TMPPY=%TEMP%\codex_verify_%RANDOM%.py"
>"%TMPPY%" echo import sys,importlib
>>"%TMPPY%" echo mods=%22torch torchvision diffusers gradio fastapi huggingface_hub numpy pydantic%22.split()
>>"%TMPPY%" echo bad=[]
>>"%TMPPY%" echo print(%22[Verify] Core libraries%22)
>>"%TMPPY%" echo for m in mods:
>>"%TMPPY%" echo ^    try:
>>"%TMPPY%" echo ^        mod=importlib.import_module(m);ver=getattr(mod,%22__version__%22,%22unknown%22);print(%22[OK]%22,m,ver)
>>"%TMPPY%" echo ^    except Exception as e:
>>"%TMPPY%" echo ^        print(%22[MISS]%22,m,e);bad.append(m)
>>"%TMPPY%" echo sys.exit(1 if bad else 0)

python "%TMPPY%"
set ERR=%ERRORLEVEL%
del /q "%TMPPY%" >NUL 2>&1
if not %ERR%==0 (
  echo [ERROR] One or more core libraries are missing. Check the log above.
  exit /b 1
)

REM 7) Optional tools
echo [6/7] Checking optional tools (ffmpeg) ...
where ffmpeg >NUL 2>&1 && echo [OK] ffmpeg found ^(video export ready^) || echo [WARN] ffmpeg not found ^(video export disabled^)

REM 8) Launch WebUI
echo [7/7] Starting WebUI ...
echo ---------------------------------------------------
python webui.py %*
set RUNERR=%ERRORLEVEL%
if not %RUNERR%==0 (
  echo [ERROR] WebUI exited with code %RUNERR%.
  popd
  exit /b %RUNERR%
)

popd
exit /b 0

