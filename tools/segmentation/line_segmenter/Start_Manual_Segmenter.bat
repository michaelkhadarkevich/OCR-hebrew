@echo off
setlocal EnableExtensions
title Manual Segmenter - PDF Portable v4
cd /d "%~dp0"

echo Manual Segmenter - PDF Portable v4
echo This test build opens at http://localhost:8770
echo.

set "PYEXE="
where py >nul 2>nul
if not errorlevel 1 set "PYEXE=py -3"
if not defined PYEXE (
  where python >nul 2>nul
  if not errorlevel 1 set "PYEXE=python"
)

if not defined PYEXE (
  echo Python 3 was not found on this computer.
  echo Install Python 3 first, then run this file again.
  pause
  exit /b 1
)

%PYEXE% -c "import fitz" >nul 2>nul
if errorlevel 1 (
  echo Installing local PDF support ^(PyMuPDF^)...
  %PYEXE% -m pip install --user PyMuPDF
  if errorlevel 1 (
    echo.
    echo Could not install PyMuPDF. Nothing else was changed.
    pause
    exit /b 1
  )
)

echo Starting Manual Segmenter...
%PYEXE% "%~dp0server.py"
if errorlevel 1 (
  echo.
  echo Manual Segmenter stopped with an error.
  pause
)
