@echo off
REM 训练所有模型的示例脚本
REM 用法: train_all_models.bat [single|mixed] [数据集名]
REM 例如: train_all_models.bat mixed
REM 例如: train_all_models.bat single UTD

cd /d %~dp0..
call conda activate gnss_ml

set MODE=%1
set DATASET=%2

if "%MODE%"=="" set MODE=mixed
if "%DATASET%"=="" set DATASET=UTD

echo.
echo ========================================
echo 训练所有模型 (LightGBM + CNN + LSTM)
echo ========================================
echo 数据模式: %MODE%
if "%MODE%"=="single" (
    echo 数据集: %DATASET%
) else (
    echo 数据集: UTD + MCD + TGD + TGS
)
echo ========================================
echo.

if "%MODE%"=="single" (
    python src\train.py --model all --mode single --dataset %DATASET% --batch_size 64 --num_workers 0
) else (
    python src\train.py --model all --mode mixed --batch_size 64 --num_workers 0
)

echo.
echo 所有模型训练完成！
pause
