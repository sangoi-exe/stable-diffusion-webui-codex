@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >NUL 2>&1
title Stable Diffusion WebUI - Codex Launcher (verify-only)

REM Load optional user settings
if exist webui.settings.bat (
    call webui.settings.bat
)

if not defined PYTHON (set PYTHON=python)

REM Resolve VENV_DIR by detection only; do NOT create here
if not defined VENV_DIR (
  if exist "%~dp0venv\Scripts\Python.exe" (
    set "VENV_DIR=%~dp0venv"
  ) else (
    if exist "%~dp0.venv\Scripts\Python.exe" (
      set "VENV_DIR=%~dp0.venv"
    ) else (
      set "VENV_DIR=%~dp0venv"
    )
  )
)

echo.
echo [Codex] Stable Diffusion WebUI launcher (verify-only)
echo ----------------------------------------------------
if defined CODEX_LOG_LEVEL (
  echo [log] CODEX_LOG_LEVEL=%CODEX_LOG_LEVEL%
) else (
  echo [log] CODEX_LOG_LEVEL not set; defaulting to DEBUG
)

REM 1) Check Python availability
echo [1/4] Checking Python...
mkdir tmp 2>NUL
%PYTHON% -c "" >tmp\stdout.txt 2>tmp\stderr.txt
if %ERRORLEVEL% == 0 goto :pyver
echo Couldn't launch python
goto :show_stdout_stderr

:pyver
REM 2) Python version (recommended 3.12)
set "_TMPVER=%TEMP%\codex_pyver_%RANDOM%.txt"
%PYTHON% -c "import sys;print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))" > "%_TMPVER%" 2>NUL
set /p PYVER=<"%_TMPVER%"
del /q "%_TMPVER%" >NUL 2>&1
if not "%PYVER%"=="3.12" (
  echo [WARN] Detected Python %PYVER%. Recommended: Python 3.12.x
)

REM 3) Activate existing venv (required)
if ["%VENV_DIR%"] == ["-"] goto :no_venv
if ["%SKIP_VENV%"] == ["1"] goto :no_venv

dir "%VENV_DIR%\Scripts\Python.exe" >NUL 2>&1
if %ERRORLEVEL% NEQ 0 goto :no_venv

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate venv at %VENV_DIR%
  echo         Run install-webui.bat to create and setup the environment.
  exit /b 1
)
set PYTHON="%VENV_DIR%\Scripts\Python.exe"

REM 4) Verify core libraries only (no installation here)
echo [2/4] Verifying core libraries ...
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
  echo [ERROR] Missing or broken dependencies. Please run install-webui.bat
  exit /b 1
)

REM Optional tools check (no install)
echo [3/4] Checking optional tools (ffmpeg) ...
where ffmpeg >NUL 2>&1 && echo [OK] ffmpeg found || echo [WARN] ffmpeg not found (video export disabled)

REM Launch WebUI
echo [4/4] Starting WebUI ...
%PYTHON% launch.py %*
if exist tmp\restart exit /b 0
exit /b %ERRORLEVEL%

:no_venv
echo [ERROR] Python venv not found at %VENV_DIR%\Scripts\Python.exe
echo         Please run install-webui.bat to set up the environment.
exit /b 1

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
