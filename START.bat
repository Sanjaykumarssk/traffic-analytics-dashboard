@echo off
REM ========================================
REM  Traffic Dashboard - One-Click Start
REM  Just double-click this file!
REM ========================================

title Traffic Dashboard - Starting...

REM Change to script directory
cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python not found!
    echo Please install Python 3.11+ from python.org
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Traffic Dashboard - Auto Starting
echo ========================================
echo.
echo Please wait while everything is set up...
echo.

REM Run automated startup
python start_server.py

REM If error, pause to see message
if errorlevel 1 (
    echo.
    echo [ERROR] Startup failed. Check the messages above.
    pause
)

