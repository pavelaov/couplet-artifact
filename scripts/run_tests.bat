@echo off
rem run_tests.bat -- run every test of the artifact (Windows).
rem Usage: scripts\run_tests.bat [GROUP ...]       groups: python cpp interop (default: all three)
rem   python   the 17 suites python\tests\test_*.py (the last five are the Couplet layer)
rem   cpp      the 9 programs selftest, check_vectors and test_* of cpp\build (test_ppml: Couplet layer)
rem   interop  the mixed-language runs of cpp\tests (one party in C++, one in Python)
rem Before cpp and interop: scripts\build_windows.bat and scripts\regen_circuits.bat.
rem Python: the variable PYTHON, else "python".
rem Output: one line per program on the console; test_logs\NAME.log (complete output of a program),
rem test_logs\summary.txt, and test_logs\test_log.txt (the summary followed by every log).
rem Exit status 0 iff every program ended with status 0.  Nothing is repeated automatically;
rem python tests\test_net.py asserts wall-clock times and can fail on a busy machine.
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
if not defined PYTHON set "PYTHON=python"
set "LOGDIR=%ROOT%test_logs"
set /a TOTAL=0
set /a PASSED=0
set /a FAILED=0
set "FAILED_NAMES="
set "NATIVE_CHECKED="
set "NATIVE_OK="
set "GROUPS=%*"
if "%GROUPS%"=="" set "GROUPS=python cpp interop"

for %%G in (%GROUPS%) do (
    set "KNOWN="
    for %%K in (python cpp interop) do if /i "%%G"=="%%K" set "KNOWN=1"
    if not defined KNOWN (
        echo unknown group %%G; the groups are: python cpp interop
        exit /b 2
    )
)

%PYTHON% -c "import numpy" >nul 2>nul
if errorlevel 1 (
    echo The interpreter "%PYTHON%" was not found or cannot import numpy: python -m pip install -r requirements.txt
    exit /b 2
)
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
del /q "%LOGDIR%\*.log" "%LOGDIR%\full.tmp" 2>nul
> "%LOGDIR%\summary.txt" echo name ^| exit status ^| seconds ^| result line of the output
type nul > "%LOGDIR%\full.tmp"
for /f "usebackq delims=" %%V in (`%PYTHON% -c "import sys, numpy; print(sys.version.split()[0], 'NumPy', numpy.__version__)"`) do set "PYVER=%%V"
echo logs: test_logs\
echo python: !PYVER!
call :now T_START

for %%G in (%GROUPS%) do call :group_%%G

call :now T_END
set /a T_ALL=T_END-T_START
if !T_ALL! lss 0 set /a T_ALL+=86400
echo.
echo SUMMARY: !TOTAL! programs, !PASSED! passed, !FAILED! failed (!T_ALL! s)
>> "%LOGDIR%\summary.txt" echo SUMMARY: !TOTAL! programs, !PASSED! passed, !FAILED! failed (!T_ALL! s)
set "STATUS=0"
if not "!FAILED!"=="0" (
    echo failed:!FAILED_NAMES!
    >> "%LOGDIR%\summary.txt" echo failed:!FAILED_NAMES!
    set "STATUS=1"
) else (
    echo ALL TESTS PASSED
)
> "%LOGDIR%\test_log.txt" echo Test log of scripts\run_tests.bat (groups: %GROUPS%)
>> "%LOGDIR%\test_log.txt" echo python: !PYVER!
>> "%LOGDIR%\test_log.txt" echo.
type "%LOGDIR%\summary.txt" >> "%LOGDIR%\test_log.txt"
type "%LOGDIR%\full.tmp" >> "%LOGDIR%\test_log.txt"
del /q "%LOGDIR%\full.tmp" 2>nul
exit /b !STATUS!

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

rem ------------------------------------------------------------------ helpers
:need_native
rem The native programs and the circuit files that they load must exist (checked once).
if defined NATIVE_CHECKED (
    if defined NATIVE_OK exit /b 0
    exit /b 1
)
set "NATIVE_CHECKED=1"
set "MISSING="
if not exist "%ROOT%cpp\build\test_protocol.exe" set "MISSING=cpp\build is not built: run scripts\build_windows.bat first"
if defined MISSING goto :native_failed
%PYTHON% "%ROOT%python\bench\regen_circuits.py" --tests --check > "%LOGDIR%\circuits_check.txt" 2>&1
if errorlevel 1 set "MISSING=the circuit files of the tests are missing or differ: run scripts\regen_circuits.bat first"
if defined MISSING goto :native_failed
set "NATIVE_OK=1"
exit /b 0
:native_failed
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
>> "%LOGDIR%\full.tmp" echo.
>> "%LOGDIR%\full.tmp" echo ===== %T_NAME% (in %T_DIR%: %T_CMD%): exit status !RC!, !ELAPSED! s
type "%LOG%" >> "%LOGDIR%\full.tmp"
goto :eof
