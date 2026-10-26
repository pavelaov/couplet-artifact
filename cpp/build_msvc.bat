@echo off
rem build_msvc.bat -- build the couplet C++ tests, benchmarks and drivers with MSVC (64-bit).
rem Usage: build_msvc.bat            (from any directory)
rem The script needs the 64-bit environment of the compiler.  It takes, in this order: the
rem environment of the calling shell if that already is an x64 developer prompt; the script named
rem by the variable VCVARS64 (full path of vcvars64.bat); the newest installation with the C++
rem tools that vswhere.exe of the Visual Studio installer reports.
rem Every target is deleted before it is compiled and must exist afterwards, so that a compiler
rem that ends abnormally cannot leave a stale executable behind a BUILD OK.
setlocal
if /i "%VSCMD_ARG_TGT_ARCH%"=="x64" goto :env_ready
if defined VCVARS64 goto :call_vcvars
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :no_msvc
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS64=%%i\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS64 goto :no_msvc
:call_vcvars
if not exist "%VCVARS64%" goto :no_msvc
call "%VCVARS64%" >nul
if errorlevel 1 (
    echo vcvars64.bat failed
    exit /b 1
)
:env_ready
cd /d "%~dp0"
if not exist build mkdir build

set CXXFLAGS=/nologo /O2 /std:c++17 /EHsc /arch:AVX2 /W4 /DNDEBUG /Iinclude

echo [cl] selftest
if exist build\selftest.exe del /q build\selftest.exe
cl %CXXFLAGS% src\selftest.cpp /Fobuild\ /Febuild\selftest.exe ws2_32.lib
if errorlevel 1 exit /b 1
if not exist build\selftest.exe goto :notbuilt

echo [cl] bench_prims
if exist build\bench_prims.exe del /q build\bench_prims.exe
cl %CXXFLAGS% src\bench_prims.cpp /Fobuild\ /Febuild\bench_prims.exe
if errorlevel 1 exit /b 1
if not exist build\bench_prims.exe goto :notbuilt

if exist src\check_vectors.cpp (
    echo [cl] check_vectors
    if exist build\check_vectors.exe del /q build\check_vectors.exe
    cl %CXXFLAGS% src\check_vectors.cpp /Fobuild\ /Febuild\check_vectors.exe
    if errorlevel 1 exit /b 1
    if not exist build\check_vectors.exe goto :notbuilt
)

if exist src\test_backend_prims.cpp (
    echo [cl] test_backend_prims
    if exist build\test_backend_prims.exe del /q build\test_backend_prims.exe
    cl %CXXFLAGS% src\test_backend_prims.cpp /Fobuild\ /Febuild\test_backend_prims.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\test_backend_prims.exe goto :notbuilt
)

if exist src\test_fss.cpp (
    echo [cl] test_fss
    if exist build\test_fss.exe del /q build\test_fss.exe
    cl %CXXFLAGS% src\test_fss.cpp /Fobuild\ /Febuild\test_fss.exe
    if errorlevel 1 exit /b 1
    if not exist build\test_fss.exe goto :notbuilt
)

if exist src\bench_fss.cpp (
    echo [cl] bench_fss
    if exist build\bench_fss.exe del /q build\bench_fss.exe
    cl %CXXFLAGS% src\bench_fss.cpp /Fobuild\ /Febuild\bench_fss.exe
    if errorlevel 1 exit /b 1
    if not exist build\bench_fss.exe goto :notbuilt
)

if exist src\test_prep.cpp (
    echo [cl] test_prep
    if exist build\test_prep.exe del /q build\test_prep.exe
    cl %CXXFLAGS% src\test_prep.cpp /Fobuild\ /Febuild\test_prep.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\test_prep.exe goto :notbuilt
)

if exist src\bench_prep.cpp (
    echo [cl] bench_prep
    if exist build\bench_prep.exe del /q build\bench_prep.exe
    cl %CXXFLAGS% src\bench_prep.cpp /Fobuild\ /Febuild\bench_prep.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\bench_prep.exe goto :notbuilt
)

if exist src\test_engine.cpp (
    echo [cl] test_engine
    if exist build\test_engine.exe del /q build\test_engine.exe
    cl %CXXFLAGS% src\test_engine.cpp /Fobuild\ /Febuild\test_engine.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\test_engine.exe goto :notbuilt
)

if exist src\test_protocol.cpp (
    echo [cl] test_protocol
    if exist build\test_protocol.exe del /q build\test_protocol.exe
    cl %CXXFLAGS% src\test_protocol.cpp /Fobuild\ /Febuild\test_protocol.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\test_protocol.exe goto :notbuilt
)

if exist src\couplet_party.cpp (
    echo [cl] couplet_party
    if exist build\couplet_party.exe del /q build\couplet_party.exe
    cl %CXXFLAGS% src\couplet_party.cpp /Fobuild\ /Febuild\couplet_party.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\couplet_party.exe goto :notbuilt
)

if exist src\test_baselines.cpp (
    echo [cl] test_baselines
    if exist build\test_baselines.exe del /q build\test_baselines.exe
    cl %CXXFLAGS% src\test_baselines.cpp /Fobuild\ /Febuild\test_baselines.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\test_baselines.exe goto :notbuilt
)

if exist src\bench_baselines.cpp (
    echo [cl] bench_baselines
    if exist build\bench_baselines.exe del /q build\bench_baselines.exe
    cl %CXXFLAGS% src\bench_baselines.cpp /Fobuild\ /Febuild\bench_baselines.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\bench_baselines.exe goto :notbuilt
)

if exist src\app_e2e.cpp (
    echo [cl] app_e2e
    if exist build\app_e2e.exe del /q build\app_e2e.exe
    cl %CXXFLAGS% src\app_e2e.cpp /Fobuild\ /Febuild\app_e2e.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\app_e2e.exe goto :notbuilt
)

if exist src\bench_gates.cpp (
    echo [cl] bench_gates
    if exist build\bench_gates.exe del /q build\bench_gates.exe
    cl %CXXFLAGS% src\bench_gates.cpp /Fobuild\ /Febuild\bench_gates.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\bench_gates.exe goto :notbuilt
)

if exist src\test_ppml.cpp (
    echo [cl] test_ppml
    if exist build\test_ppml.exe del /q build\test_ppml.exe
    cl %CXXFLAGS% src\test_ppml.cpp /Fobuild\ /Febuild\test_ppml.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\test_ppml.exe goto :notbuilt
)

if exist src\couplet_infer.cpp (
    echo [cl] couplet_infer
    if exist build\couplet_infer.exe del /q build\couplet_infer.exe
    cl %CXXFLAGS% src\couplet_infer.cpp /Fobuild\ /Febuild\couplet_infer.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\couplet_infer.exe goto :notbuilt
)

if exist src\bench_ppml.cpp (
    echo [cl] bench_ppml
    if exist build\bench_ppml.exe del /q build\bench_ppml.exe
    cl %CXXFLAGS% src\bench_ppml.cpp /Fobuild\ /Febuild\bench_ppml.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\bench_ppml.exe goto :notbuilt
)

if exist src\couplet_item.cpp (
    echo [cl] couplet_item
    if exist build\couplet_item.exe del /q build\couplet_item.exe
    cl %CXXFLAGS% src\couplet_item.cpp /Fobuild\ /Febuild\couplet_item.exe ws2_32.lib bcrypt.lib
    if errorlevel 1 exit /b 1
    if not exist build\couplet_item.exe goto :notbuilt
)

echo BUILD OK
endlocal
exit /b 0

:notbuilt
rem The compiler ended without an executable although it did not report an error level of 1 or more
rem (seen once: an internal compiler error, C1001, after which the script went on).
echo BUILD FAILED: the last target was not produced
endlocal
exit /b 1

:no_msvc
echo MSVC not found: run this script from an x64 Native Tools Command Prompt, or set VCVARS64 to
echo the full path of vcvars64.bat of the installation.
endlocal
exit /b 1
