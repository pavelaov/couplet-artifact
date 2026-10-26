@echo off
rem run_experiments.bat -- the measurement campaign (python\bench\run_campaign.py).
rem Usage: scripts\run_experiments.bat --quick      smoke run of every driver (minutes), output in
rem                                                  experiments\quick\campaign
rem        scripts\run_experiments.bat [ARGS...]     the whole campaign (about 8 hours on the authors'
rem                                                  machine; the largest rows need 36 GiB of memory
rem                                                  per party), output in
rem                                                  experiments\raw\campaign
rem Further ARGS go to run_campaign.py (for example --only u,k or --match mlp16; see --help).
rem The whole campaign first rebuilds every circuit of circuits\manifest.json that is missing (3.6 GB).
rem Then: scripts\make_tables.bat quick   or   scripts\make_tables.bat campaign
rem Python: the variable PYTHON, else "python".
setlocal EnableExtensions
for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
if not defined PYTHON set "PYTHON=python"
%PYTHON% -c "import numpy" >nul 2>nul
if errorlevel 1 (
    echo run_experiments.bat: "%PYTHON%" cannot import numpy: python -m pip install -r requirements.txt
    exit /b 2
)
for %%P in (couplet_item couplet_party couplet_infer bench_ppml bench_baselines) do (
    if not exist "%ROOT%cpp\build\%%P.exe" (
        echo run_experiments.bat: cpp\build\%%P.exe is missing: run scripts\build_windows.bat first
        exit /b 2
    )
)
set "QUICK="
set "NOREGEN="
for %%A in (%*) do (
    if /i "%%~A"=="--quick" set "QUICK=1"
    if /i "%%~A"=="--list" set "NOREGEN=1"
    if /i "%%~A"=="--summary-only" set "NOREGEN=1"
    if /i "%%~A"=="--help" set "NOREGEN=1"
    if /i "%%~A"=="-h" set "NOREGEN=1"
)
if not defined QUICK if not defined NOREGEN (
    %PYTHON% "%ROOT%python\bench\regen_circuits.py" --no-bristol
    if errorlevel 1 (
        echo run_experiments.bat: the circuit files could not be rebuilt
        exit /b 1
    )
)
pushd "%ROOT%python"
%PYTHON% bench\run_campaign.py %*
set "RC=%errorlevel%"
popd
endlocal & exit /b %RC%
