@echo off
REM 训练混合数据集的示例脚本
REM 用法: train_mixed_dataset.bat [模型名]
REM 例如: train_mixed_dataset.bat lstm

cd /d %~dp0..
call conda activate gnss_ml

set MODEL=%1

if "%MODEL%"=="" set MODEL=lightgbm

echo.
echo ========================================
echo 训练混合数据集 (UTD + MCD + TGD + TGS)
echo ========================================
echo 模型: %MODEL%
echo ========================================
echo.

python src\train.py --model %MODEL% --mode mixed --batch_size 64 --num_workers 0

echo.
echo 训练完成！
pause
