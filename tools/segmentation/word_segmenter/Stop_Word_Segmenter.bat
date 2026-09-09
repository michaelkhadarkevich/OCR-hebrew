@echo off
setlocal
cd /d "%~dp0"
if not exist word_segmenter.pid (
  echo Word Segmenter PID file was not found.
  pause
  exit /b 0
)
set /p PID=<word_segmenter.pid
if "%PID%"=="" exit /b 0
taskkill /PID %PID% /F >nul 2>nul
if exist word_segmenter.pid del /q word_segmenter.pid >nul 2>nul
echo Word Segmenter stopped.
timeout /t 1 >nul
