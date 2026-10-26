@echo off
rem regen_circuits.bat -- rebuild the circuit files of circuits\ from circuits\manifest.json and check
rem their SHA-256 digests against it (the files are not shipped; the compiler is deterministic).
rem Usage: scripts\regen_circuits.bat            the 36 circuits that the tests load (about 0.4 GB, < 1 min)
rem        scripts\regen_circuits.bat --missing  every circuit of the manifest that is missing (3.6 GB)
rem        scripts\regen_circuits.bat --check    verify the files that are present
rem        scripts\regen_circuits.bat NAME ...   the named circuits
rem The Bristol text versions are deleted after their digest was checked (no program reads them).
rem Python: the variable PYTHON, else "python".
setlocal
if not defined PYTHON set "PYTHON=python"
%PYTHON% -c "import numpy" >nul 2>nul
if errorlevel 1 (
    echo regen_circuits.bat: "%PYTHON%" cannot import numpy: python -m pip install -r requirements.txt
    exit /b 2
)
set "ARGS=%*"
if "%~1"=="" set "ARGS=--tests"
if /i "%~1"=="--missing" set "ARGS="
%PYTHON% "%~dp0..\python\bench\regen_circuits.py" --no-bristol %ARGS%
set "RC=%errorlevel%"
endlocal & exit /b %RC%
