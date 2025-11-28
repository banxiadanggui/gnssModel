@echo off
REM GNSS干扰检测项目环境配置脚本
REM 自动创建conda环境并安装所有依赖

echo.
echo ========================================
echo GNSS干扰检测项目环境配置
echo ========================================
echo.

REM 检查conda是否已安装
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到conda命令
    echo 请先安装Miniconda或Anaconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/4] 检查conda版本...
conda --version

echo.
echo [2/4] 创建conda环境 'gnss_ml'...
conda env list | findstr /C:"gnss_ml" >nul
if %ERRORLEVEL% EQU 0 (
    echo 环境 'gnss_ml' 已存在
    choice /C YN /M "是否删除现有环境并重新创建"
    if errorlevel 2 goto skip_create
    if errorlevel 1 (
        echo 删除现有环境...
        call conda env remove -n gnss_ml -y
    )
)

echo 创建新环境...
call conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 环境创建失败
    echo 尝试使用备用方法...
    call conda create -n gnss_ml python=3.10 -y
    call conda activate gnss_ml
    pip install -r requirements.txt
)

:skip_create
echo.
echo [3/4] 激活环境...
call conda activate gnss_ml

echo.
echo [4/4] 验证安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import lightgbm; print(f'LightGBM版本: {lightgbm.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas版本: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn版本: {sklearn.__version__}')"

echo.
echo ========================================
echo 环境配置完成！
echo ========================================
echo.
echo 使用方法:
echo 1. 激活环境: conda activate gnss_ml
echo 2. 训练模型: python src\train.py --model lightgbm --mode mixed
echo 3. 评估模型: python src\evaluate.py --model lightgbm --mode mixed
echo.
echo 详细使用说明请参考:
echo - README.md
echo - scripts\TRAINING_GUIDE.md
echo.
pause
