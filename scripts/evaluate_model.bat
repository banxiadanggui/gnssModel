@echo off
REM 评估模型的示例脚本
REM 用法: evaluate_model.bat [模型名] [single|mixed] [数据集名]
REM 例如: evaluate_model.bat cnn mixed
REM 例如: evaluate_model.bat lstm single UTD

cd /d %~dp0..
call conda activate gnss_ml

set MODEL=%1
set MODE=%2
set DATASET=%3

if "%MODEL%"=="" set MODEL=lightgbm
if "%MODE%"=="" set MODE=mixed
if "%DATASET%"=="" set DATASET=UTD

echo.
echo ========================================
echo 评估模型
echo ========================================
echo 模型: %MODEL%
echo 数据模式: %MODE%
if "%MODE%"=="single" (
    echo 数据集: %DATASET%
) else (
    echo 数据集: UTD + MCD + TGD + TGS
)
echo ========================================
echo.

if "%MODE%"=="single" (
    python src\evaluate.py --model %MODEL% --mode single --dataset %DATASET% --batch_size 64 --num_workers 0 --save_predictions
) else (
    python src\evaluate.py --model %MODEL% --mode mixed --batch_size 64 --num_workers 0 --save_predictions
)

echo.
echo 评估完成！
pause
