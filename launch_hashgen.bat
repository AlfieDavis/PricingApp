@echo off
setlocal
REM This batch script launches the generate_hash.py script from the current directory.
REM It prefers the project's virtual environment Python if it exists.

REM Change to the directory where this script is located
cd /d "%~dp0"

REM Default Python interpreter name
set "PY=python"

REM Prefer Python inside a 'venv' directory if it exists
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"

"%PY%" generate_hash.py
endlocal