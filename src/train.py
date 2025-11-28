"""
训练主脚本
支持三种模型(LightGBM/CNN/LSTM)和两种数据模式(single/mixed)
"""
import argparse
import os
import sys
sys.path.append(os.path.dirname(__file__))

import config
from dataset import prepare_data, create_dataloaders
from models.lightgbm_model import train_lightgbm
from models.cnn_model import train_cnn
from models.lstm_model import train_lstm
from utils import set_seed

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GNSS干扰检测模型训练')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['lightgbm', 'cnn', 'lstm', 'all'],
        help='选择训练的模型: lightgbm, cnn, lstm, 或 all (训练所有模型)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='mixed',
        choices=['single', 'mixed'],
        help='数据模式: single (单个数据集) 或 mixed (混合所有数据集)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='UTD',
        choices=['UTD', 'MCD', 'TGD', 'TGS'],
        help='如果mode=single，选择使用的数据集'
    )

    parser.add_argument(
        '--normalize',
        type=str,
        default='standard',
        choices=['standard', 'minmax', 'none'],
        help='归一化方法: standard, minmax, 或 none'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='批次大小 (仅用于CNN和LSTM)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='数据加载器的工作进程数 (Windows建议设为0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    return parser.parse_args()

def train_model(model_name, data_dict, train_loader, val_loader, test_loader,
                dataset_mode, single_dataset):
    """
    训练指定模型

    Args:
        model_name: 模型名称 ('lightgbm', 'cnn', 'lstm')
        data_dict: 包含numpy数据的字典 (用于LightGBM)
        train_loader, val_loader, test_loader: DataLoader (用于CNN和LSTM)
        dataset_mode: 'single' 或 'mixed'
        single_dataset: 如果是single模式，数据集名称

    Returns:
        训练好的模型和评估结果
    """
    if model_name == 'lightgbm':
        print("\n" + "="*80)
        print("训练 LightGBM 模型")
        print("="*80)
        model, results = train_lightgbm(
            data_dict,
            dataset_mode=dataset_mode,
            single_dataset=single_dataset
        )

    elif model_name == 'cnn':
        print("\n" + "="*80)
        print("训练 CNN 模型")
        print("="*80)
        model, results = train_cnn(
            train_loader, val_loader, test_loader,
            dataset_mode=dataset_mode,
            single_dataset=single_dataset
        )

    elif model_name == 'lstm':
        print("\n" + "="*80)
        print("训练 LSTM 模型")
        print("="*80)
        model, results = train_lstm(
            train_loader, val_loader, test_loader,
            dataset_mode=dataset_mode,
            single_dataset=single_dataset
        )

    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    return model, results

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 打印配置信息
    print("\n" + "="*80)
    print("训练配置")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"数据模式: {args.mode}")
    if args.mode == 'single':
        print(f"数据集: {args.dataset}")
    else:
        print(f"数据集: {', '.join(config.DATASETS)}")
    print(f"归一化: {args.normalize}")
    print(f"批次大小: {args.batch_size}")
    print(f"随机种子: {args.seed}")
    print("="*80)

    # 准备数据
    normalize = args.normalize != 'none'
    data_dict = prepare_data(
        dataset_mode=args.mode,
        single_dataset=args.dataset,
        normalize=normalize,
        normalization_method=args.normalize if normalize else 'standard'
    )

    # 创建DataLoader (用于PyTorch模型)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict['X_train'],
        data_dict['X_val'],
        data_dict['X_test'],
        data_dict['y_train'],
        data_dict['y_val'],
        data_dict['y_test'],
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 训练模型
    if args.model == 'all':
        # 训练所有模型
        print("\n" + "="*80)
        print("将依次训练所有模型: LightGBM -> CNN -> LSTM")
        print("="*80)

        all_results = {}

        for model_name in ['lightgbm', 'cnn', 'lstm']:
            model, results = train_model(
                model_name,
                data_dict,
                train_loader, val_loader, test_loader,
                args.mode,
                args.dataset
            )
            all_results[model_name] = results

        # 打印所有模型的对比结果
        print("\n" + "="*80)
        print("所有模型测试集结果对比")
        print("="*80)
        print(f"{'模型':<15} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12}")
        print("-"*80)
        for model_name in ['lightgbm', 'cnn', 'lstm']:
            metrics = all_results[model_name]['test_metrics']
            print(f"{model_name.upper():<15} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1']:<12.4f}")
        print("="*80 + "\n")

    else:
        # 训练单个模型
        model, results = train_model(
            args.model,
            data_dict,
            train_loader, val_loader, test_loader,
            args.mode,
            args.dataset
        )

    print("\n训练完成!")
    print(f"模型和结果已保存至:")
    print(f"  模型: {config.MODELS_DIR}")
    print(f"  图表: {config.FIGURES_DIR}")
    print(f"  日志: {config.LOGS_DIR}\n")

if __name__ == '__main__':
    main()
