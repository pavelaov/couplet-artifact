@echo off
rem build_windows.bat -- build the 19 native programs into cpp\build with MSVC (x64, C++17, /O2 /arch:AVX2).
rem Usage: scripts\build_windows.bat      (Visual Studio 2019 or newer with the C++ x64 tools; the
rem script finds the compiler itself, see cpp\build_msvc.bat; VCVARS64 may name vcvars64.bat)
setlocal
call "%~dp0..\cpp\build_msvc.bat"
if errorlevel 1 (
    echo build_windows.bat: the build failed
    exit /b 1
)
endlocal
exit /b 0
