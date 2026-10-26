@echo off
rem make_tables.bat -- generate the tables (LaTeX bodies, number macros, PROVENANCE.md) and the figures
rem (PDF) of the evaluation from a campaign summary, into out\tables and out\figures.
rem Usage: scripts\make_tables.bat [results|campaign|quick|PATH\summary.json]
rem   results   (default) results\summary.json, the shipped outputs; the generated files are then
rem             compared with results\tables and results\figures (exit status 1 if a table differs)
rem   campaign  experiments\raw\campaign\summary.json (scripts\run_experiments.bat)
rem   quick     experiments\quick\campaign\summary.json (scripts\run_experiments.bat --quick)
rem Configurations that a summary does not contain are left out and listed in out\tables\PROVENANCE.md.
rem Python: the variable PYTHON, else "python" (needs numpy and matplotlib).
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
if not defined PYTHON set "PYTHON=python"
%PYTHON% -c "import numpy, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo make_tables.bat: "%PYTHON%" cannot import numpy and matplotlib: python -m pip install -r requirements.txt
    exit /b 2
)
set "SRC=%~1"
if "%SRC%"=="" set "SRC=results"
set "SUMMARY=%SRC%"
if /i "%SRC%"=="results" set "SUMMARY=%ROOT%results\summary.json"
if /i "%SRC%"=="campaign" set "SUMMARY=%ROOT%experiments\raw\campaign\summary.json"
if /i "%SRC%"=="quick" set "SUMMARY=%ROOT%experiments\quick\campaign\summary.json"
if not exist "%SUMMARY%" (
    echo make_tables.bat: no summary at %SUMMARY%
    exit /b 2
)
for %%S in ("%SUMMARY%") do set "SUMMARY=%%~fS"
set "OUT=%ROOT%out"
if exist "%OUT%\tables" rmdir /s /q "%OUT%\tables"
if exist "%OUT%\figures" rmdir /s /q "%OUT%\figures"
pushd "%ROOT%python"
%PYTHON% bench\make_tables.py --summary "%SUMMARY%" --out "%OUT%\tables"
if errorlevel 1 (
    popd
    exit /b 1
)
%PYTHON% bench\make_figures.py --summary "%SUMMARY%" --out "%OUT%\figures"
if errorlevel 1 (
    popd
    exit /b 1
)
popd
if /i not "%SRC%"=="results" exit /b 0

rem Tables must be reproduced byte for byte.  The PDF files embed the matplotlib version and its font
rem subsets, so they are equal only with the matplotlib of the recorded run (3.11.2).
set /a DIFFER=0
for %%F in ("%ROOT%results\tables\*") do (
    fc /b "%%~fF" "%OUT%\tables\%%~nxF" >nul 2>nul
    if errorlevel 1 (
        echo DIFFERS  results\tables\%%~nxF
        set /a DIFFER+=1
    ) else (
        echo same     results\tables\%%~nxF
    )
)
for %%F in ("%ROOT%results\figures\*") do (
    fc /b "%%~fF" "%OUT%\figures\%%~nxF" >nul 2>nul
    if errorlevel 1 (
        echo differs  results\figures\%%~nxF ^(byte-identical only with matplotlib 3.11.2; compare visually^)
    ) else (
        echo same     results\figures\%%~nxF
    )
)
if not "!DIFFER!"=="0" (
    echo !DIFFER! table files differ from results\tables
    exit /b 1
)
echo every table file of results\tables was reproduced byte for byte in out\tables
exit /b 0
