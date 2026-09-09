@echo off
setlocal EnableExtensions
title Word Segmenter - Auto v3 Punctuation Safe
cd /d "%~dp0"

echo Word Segmenter - Auto v3 Punctuation Safe
echo Opens at http://localhost:8773
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
  echo Installing local PDF support ^(PyMuPDF^) ...
  %PYEXE% -m pip install --user PyMuPDF
  if errorlevel 1 (
    echo Could not install PyMuPDF.
    pause
    exit /b 1
  )
)

echo Starting Word Segmenter...
%PYEXE% "%~dp0server.py"
if errorlevel 1 (
  echo.
  echo Word Segmenter stopped with an error.
  pause
)
