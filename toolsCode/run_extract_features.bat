@echo off
echo Running MATLAB feature extraction script...
echo.

REM 检查 MATLAB 是否存在
if exist "D:\software\matlab\bin\matlab.exe" (
    "D:\software\matlab\bin\matlab.exe" -batch "cd('D:\skill\beidou\toolsCode'); extract_features; exit"
) else (
    echo Error: MATLAB not found at D:\software\matlab\bin\matlab.exe
    echo Please modify the path in this script to point to your MATLAB installation
    pause
    exit /b 1
)

echo.
echo Feature extraction completed!
pause
