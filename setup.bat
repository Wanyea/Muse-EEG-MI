@echo off
setlocal enabledelayedexpansion

REM ===========================================
REM Muse MI Trainer - Setup
REM - Creates .venv if it doesn't exist
REM - Installs requirements
REM ===========================================

cd /d "%~dp0"

set VENV_DIR=.venv
set PY_EXE=python

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [setup] Creating virtual environment: %VENV_DIR%
    %PY_EXE% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [setup][ERROR] Failed to create venv.
        pause
        exit /b 1
    )
) else (
    echo [setup] Virtual environment already exists: %VENV_DIR%
)

echo [setup] Activating venv...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [setup][ERROR] Failed to activate venv.
    pause
    exit /b 1
)

echo [setup] Upgrading pip...
python -m pip install --upgrade pip

if not exist "requirements.txt" (
    echo [setup][ERROR] requirements.txt not found in: %cd%
    pause
    exit /b 1
)

echo [setup] Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [setup][ERROR] pip install failed.
    pause
    exit /b 1
)

echo.
echo [setup] Done!
echo [setup] Next: run run_app.bat
echo.
pause
endlocal