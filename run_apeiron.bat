@echo off
setlocal enabledelayedexpansion
title Apeiron GPU/CPU Fractal Explorer
echo Starting Apeiron...

:: 1. Detect Anaconda / Python executable
set "PYTHON_EXE="

if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
    goto :found_python
)
if exist "C:\Users\szabo\anaconda3\python.exe" (
    set "PYTHON_EXE=C:\Users\szabo\anaconda3\python.exe"
    goto :found_python
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
    goto :found_python
)
if exist "%LOCALAPPDATA%\anaconda3\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\anaconda3\python.exe"
    goto :found_python
)
if exist "C:\ProgramData\anaconda3\python.exe" (
    set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
    goto :found_python
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_EXE=py"
    goto :found_python
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_EXE=python"
    goto :found_python
)

:found_python
if "%PYTHON_EXE%"=="" (
    echo.
    echo [ERROR] Python / Anaconda was not located automatically.
    echo Please launch Apeiron from your Anaconda Prompt:
    echo   conda activate base
    echo   python apeiron.py
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0apeiron.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Apeiron exited with an error.
    pause
)
