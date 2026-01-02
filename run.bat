@echo off
setlocal

REM ===========================================
REM Muse MI Trainer - Run
REM - Activates .venv
REM - Starts server.py with hardcoded BLE address
REM ===========================================

cd /d "%~dp0"

set VENV_DIR=.venv

set MUSE_ADDR=00:55:DA:B9:F0:5E

REM Preset (keep p1041 unless you know otherwise)
set PRESET=p1041

REM Optional ports
set WS_PORT=5115
set OSC_IP=127.0.0.1
set OSC_PORT=5000

if "%MUSE_ADDR%"=="PASTE_YOUR_MUSE_BLE_ADDRESS_HERE" (
    echo [run][ERROR] You must set MUSE_ADDR in run_app.bat
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [run][ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [run][ERROR] Failed to activate venv.
    pause
    exit /b 1
)

if not exist "server.py" (
    echo [run][ERROR] server.py not found in: %cd%
    pause
    exit /b 1
)

echo [run] Starting Muse MI Trainer...
echo [run] Address: %MUSE_ADDR%
echo [run] Preset:  %PRESET%
echo [run] UI:      http://localhost:%WS_PORT%
echo.

python server.py --address "%MUSE_ADDR%" --preset "%PRESET%" --ws_port %WS_PORT% --osc_ip %OSC_IP% --osc_port %OSC_PORT%
if errorlevel 1 (
    echo.
    echo [run][ERROR] server.py exited with an error.
    pause
    exit /b 1
)

endlocal