@echo off
rem Starts FDA (always listening) and opens the display in the browser.
cd /d "%~dp0\..\.."
call .venv\Scripts\activate.bat
python -m fda %*
if errorlevel 1 pause
