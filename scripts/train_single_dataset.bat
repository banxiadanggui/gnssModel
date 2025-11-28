@echo off
REM 训练单个数据集的示例脚本
REM 用法: train_single_dataset.bat [模型名] [数据集名]
REM 例如: train_single_dataset.bat lightgbm UTD

cd /d %~dp0..
call conda activate gnss_ml

set MODEL=%1
set DATASET=%2

if "%MODEL%"=="" set MODEL=lightgbm
if "%DATASET%"=="" set DATASET=UTD

echo.
echo ========================================
echo 训练单个数据集
echo ========================================
echo 模型: %MODEL%
echo 数据集: %DATASET%
echo ========================================
echo.

python src\train.py --model %MODEL% --mode single --dataset %DATASET% --batch_size 64 --num_workers 0

echo.
echo 训练完成！
pause
