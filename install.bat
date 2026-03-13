@echo off
setlocal
cd /d "%~dp0"

echo [Chatterbox] Starting Installation Sequence...

:: 1. Create venv if it doesn't exist
if not exist "venv" (
    echo [Chatterbox] Creating virtual environment...
    py -m venv venv
)

:: 2. Upgrade pip
echo [Chatterbox] Upgrading pip...
.\venv\Scripts\python.exe -m pip install --upgrade pip

:: 3. Install dependencies
echo [Chatterbox] Installing dependencies from requirements.txt...
.\venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo [Chatterbox] Installation complete!
echo.
echo [Note] If you have an RTX 50-series GPU (Blackwell), you may need to 
echo run the special hardware support command mentioned in the README.
echo.
pause
endlocal
