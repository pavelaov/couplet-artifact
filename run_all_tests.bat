@echo off
rem run_all_tests.bat -- run every test program of the artifact (Windows).
rem
rem Usage: run_all_tests.bat [GROUP ...]          (from any directory)
rem
rem Groups (default: the first three, in this order):
rem   python      the 17 suites python\tests\test_*.py of the reference implementation (the last
rem               five, test_ppml_circuits to test_ppml_attacks, are the Couplet layer)
rem   cpp         the 9 programs selftest, check_vectors and test_* of cpp\build (test_ppml is the
rem               Couplet layer)
rem   interop     the mixed-language runs (one party per language) of cpp\tests; interop_infer is
rem               the inference of the Couplet layer (couplet_infer against the reference)
rem   generators  the self-tests of the table and figure generators in python\bench.
rem               Not in the default set: they read the measurement records under
rem               experiments\raw, written by python\bench\run_campaign.py.
rem
rem Before the groups cpp and interop: build with cpp\build_msvc.bat, and make sure that the
rem circuit files exist (python python\bench\regen_circuits.py --tests; they are not part of the packaged archive).
rem The Python interpreter is "python"; set the variable PYTHON to use another one.
rem
rem Every program runs with its default arguments.  Its complete output goes to
rem test_logs\NAME.log; the console and test_logs\summary.txt get one line per program
rem (name, exit status, seconds, and the result line: the last output line that names its
rem checks, else the last line).  The exit status of this script is 0 iff every program
rem ended with status 0.  Nothing is repeated automatically: python
rem tests\test_net.py asserts wall-clock times and can fail on a busy machine; run it again
rem alone in that case.
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
if not defined PYTHON set "PYTHON=python"
set "LOGDIR=%ROOT%test_logs"
set /a TOTAL=0
set /a PASSED=0
set /a FAILED=0
set "FAILED_NAMES="
set "GROUPS=%*"
if "%GROUPS%"=="" set "GROUPS=python cpp interop"

for %%G in (%GROUPS%) do (
    set "KNOWN="
    for %%K in (python cpp interop generators) do if /i "%%G"=="%%K" set "KNOWN=1"
    if not defined KNOWN (
        echo unknown group %%G; the groups are: python cpp interop generators
        exit /b 2
    )
)

%PYTHON% -c "import numpy" >nul 2>nul
if errorlevel 1 (
    echo The interpreter "%PYTHON%" was not found or cannot import numpy; see requirements.txt.
    exit /b 2
)
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
> "%LOGDIR%\summary.txt" echo name ^| exit status ^| seconds ^| result line of the output
echo logs: %LOGDIR%

for %%G in (%GROUPS%) do call :group_%%G

echo.
echo SUMMARY: !TOTAL! programs, !PASSED! passed, !FAILED! failed
>> "%LOGDIR%\summary.txt" echo SUMMARY: !TOTAL! programs, !PASSED! passed, !FAILED! failed
if not "!FAILED!"=="0" (
    echo failed:!FAILED_NAMES!
    >> "%LOGDIR%\summary.txt" echo failed:!FAILED_NAMES!
    exit /b 1
)
echo ALL TESTS PASSED
exit /b 0

rem ------------------------------------------------------------------ groups
:group_python
for %%T in (test_abb test_attacks test_baselines test_circuits test_fss_exhaustive test_gates test_net test_prep test_primitives test_protocol test_spdz2k test_autheval test_ppml_circuits test_planes test_truncmask test_nn test_ppml_attacks) do (
    set "T_NAME=py_%%T"
    set "T_DIR=python"
    set "T_CMD=%PYTHON% tests\%%T.py"
    call :run
)
goto :eof

:group_cpp
call :need_native
if errorlevel 1 goto :eof
for %%T in (selftest check_vectors test_backend_prims test_fss test_prep test_engine test_protocol test_baselines test_ppml) do (
    set "T_NAME=cpp_%%T"
    set "T_DIR=cpp"
    set "T_CMD=build\%%T.exe"
    call :run
)
goto :eof

:group_interop
call :need_native
if errorlevel 1 goto :eof
set "T_DIR=cpp"
set "T_NAME=interop_gen"
set "T_CMD=%PYTHON% tests\interop_peer.py --spawn --all"
call :run
set "T_NAME=interop_online"
set "T_CMD=%PYTHON% tests\interop_peer.py --online --spawn --all"
call :run
set "T_NAME=interop_baselines"
set "T_CMD=%PYTHON% tests\interop_baselines.py --spawn --all"
call :run
set "T_NAME=interop_prep"
set "T_CMD=%PYTHON% tests\interop_prep.py --spawn --all"
call :run
set "T_NAME=interop_infer"
set "T_CMD=%PYTHON% tests\interop_peer.py --infer --spawn --all"
call :run
goto :eof

:group_generators
set "T_DIR=python"
set "T_NAME=gen_make_tables"
set "T_CMD=%PYTHON% bench\make_tables.py --selftest"
call :run
set "T_NAME=gen_make_figures"
set "T_CMD=%PYTHON% bench\make_figures.py --selftest"
call :run
goto :eof

rem ------------------------------------------------------------------ helpers
:need_native
rem The native programs and the circuit files that they load must exist.
set "MISSING="
if not exist "%ROOT%cpp\build\test_protocol.exe" set "MISSING=cpp\build is not built: run cpp\build_msvc.bat first"
if not defined MISSING if not exist "%ROOT%circuits\dpf_n8_k64_s64_xor_one.bin" set "MISSING=the circuit files are missing: run  python python\bench\regen_circuits.py --tests  first"
if not defined MISSING exit /b 0
if defined NATIVE_REPORTED exit /b 1
set "NATIVE_REPORTED=1"
set /a TOTAL+=1
set /a FAILED+=1
set "FAILED_NAMES=!FAILED_NAMES! native-prerequisites"
echo [!TOTAL!] native prerequisites
echo       FAIL  !MISSING!
>> "%LOGDIR%\summary.txt" echo native-prerequisites ^| 1 ^| 0 ^| !MISSING!
exit /b 1

:now
rem Seconds since midnight into the variable named by the first argument.
for /f "tokens=1-3 delims=:.," %%a in ("%time: =0%") do set /a "%~1=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)"
goto :eof

:run
rem Runs T_CMD in the directory T_DIR (relative to the artifact root) under the name T_NAME.
set /a TOTAL+=1
set "LOG=%LOGDIR%\%T_NAME%.log"
echo [!TOTAL!] %T_NAME%   (in %T_DIR%: %T_CMD%)
call :now T0
pushd "%ROOT%%T_DIR%"
%T_CMD% > "%LOG%" 2>&1
set "RC=!errorlevel!"
popd
call :now T1
set /a ELAPSED=T1-T0
if !ELAPSED! lss 0 set /a ELAPSED+=86400
rem The line to show: the last one that names its checks, else the last line of the output.
set "LAST="
for /f "usebackq delims=" %%L in (`findstr /i /c:"checks" "%LOG%"`) do set "LAST=%%L"
if not defined LAST for /f "usebackq delims=" %%L in ("%LOG%") do set "LAST=%%L"
if "!RC!"=="0" (
    set /a PASSED+=1
    echo       PASS  ^(!ELAPSED! s^)  !LAST!
) else (
    set /a FAILED+=1
    set "FAILED_NAMES=!FAILED_NAMES! %T_NAME%"
    echo       FAIL  exit status !RC!  ^(!ELAPSED! s^)  !LAST!
)
>> "%LOGDIR%\summary.txt" echo %T_NAME% ^| !RC! ^| !ELAPSED! ^| !LAST!
goto :eof
