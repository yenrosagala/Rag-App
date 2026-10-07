@echo off
REM Sets up the Laptop RAG App (Clippy edition) and builds a standalone .exe.
REM Run this once from this folder (double-click or run from cmd/PowerShell).

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install it from https://www.python.org/downloads/ first.
    echo Make sure to check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo Setting up Laptop RAG App environment...
python -m venv venv
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Building standalone Clippy.exe with PyInstaller...
pyinstaller --noconfirm --onefile --windowed --name Clippy ^
    --add-data "modules;modules" ^
    --add-data "assets;assets" ^
    --collect-all sentence_transformers ^
    --collect-all chromadb ^
    clippy_app.py

echo.
echo Build complete! Find it at dist\Clippy.exe
echo.
echo Next steps:
echo   1. Install Ollama from https://ollama.com and run:
echo        ollama pull llama3.2:3b-instruct-q4_K_M
echo   2. Add documents to the data\ folder next to Clippy.exe.
echo   3. Double-click Clippy.exe, then right-click him -^> Settings to
echo      point at your data folder and Rebuild index.
echo.
pause
