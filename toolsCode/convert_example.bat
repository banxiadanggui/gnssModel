@echo off
REM ====================================================================
REM GNSS MAT to CSV Converter - 示例批处理文件
REM ====================================================================
REM
REM 使用说明：
REM 1. 修改下方的MAT_FILE变量，指向你的MAT文件路径
REM 2. (可选) 修改OUTPUT_DIR变量，指定输出目录
REM 3. 双击运行此批处理文件
REM
REM ====================================================================

echo.
echo ====================================================================
echo GNSS MAT to CSV Converter
echo ====================================================================
echo.

REM 配置MAT文件路径（请根据实际情况修改）
set MAT_FILE=D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat

REM 配置输出目录（可选，留空则使用默认目录）
set OUTPUT_DIR=

REM ====================================================================
REM 以下内容无需修改
REM ====================================================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.6+
    echo.
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [信息] Python已安装
python --version
echo.

REM 检查必需的库
echo [信息] 检查依赖库...
python -c "import numpy, pandas, scipy" >nul 2>&1
if errorlevel 1 (
    echo [警告] 缺少必需的库，正在安装...
    pip install numpy pandas scipy h5py
    if errorlevel 1 (
        echo [错误] 安装失败，请手动运行: pip install numpy pandas scipy h5py
        pause
        exit /b 1
    )
)
echo [信息] 所有依赖库已就绪
echo.

REM 检查MAT文件是否存在
if not exist "%MAT_FILE%" (
    echo [错误] MAT文件不存在: %MAT_FILE%
    echo.
    echo 请修改此批处理文件中的 MAT_FILE 变量，指向正确的文件路径
    pause
    exit /b 1
)

echo [信息] MAT文件: %MAT_FILE%
echo.

REM 执行转换
echo [信息] 开始转换...
echo.

if "%OUTPUT_DIR%"=="" (
    python gnss_mat_to_csv.py "%MAT_FILE%"
) else (
    python gnss_mat_to_csv.py "%MAT_FILE%" "%OUTPUT_DIR%"
)

if errorlevel 1 (
    echo.
    echo [错误] 转换失败
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo 转换完成！
echo ====================================================================
echo.

pause
