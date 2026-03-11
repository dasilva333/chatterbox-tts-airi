@echo off
setlocal

:: Navigate to the script's directory
cd /d "%~dp0"

echo [Chatterbox] Starting TTS Server...

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [Error] Virtual environment not found in "venv".
    echo Please ensure you have created the venv using Python 3.11.
    pause
    exit /b 1
)

:: Run the server using the venv python directly
:: Pass all arguments (like --profile) through to server.py
.\venv\Scripts\python.exe server.py %*

if %ERRORLEVEL% neq 0 (
    echo [Error] Server stopped unexpectedly with exit code %ERRORLEVEL%.
    pause
)

endlocal
