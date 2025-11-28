"""
评估脚本
用于加载已保存的模型并在测试集上评估
"""
import argparse
import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import lightgbm as lgb

import config
from dataset import prepare_data, create_dataloaders
from models.lightgbm_model import LightGBMClassifier
from models.cnn_model import CNN1D, predict as predict_cnn
from models.lstm_model import LSTMClassifier, predict as predict_lstm
from utils import (plot_confusion_matrix, print_classification_report,
                  get_device, set_seed)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GNSS干扰检测模型评估')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['lightgbm', 'cnn', 'lstm'],
        help='选择评估的模型: lightgbm, cnn, lstm'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        help='模型文件路径 (如果不指定，将使用默认路径)'
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

    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='是否保存预测结果'
    )

    return parser.parse_args()

def evaluate_lightgbm(model_path, X_test, y_test, dataset_tag):
    """
    评估LightGBM模型

    Args:
        model_path: 模型文件路径
        X_test: 测试数据
        y_test: 测试标签
        dataset_tag: 数据集标签 (用于保存结果)

    Returns:
        预测结果和评估指标
    """
    print("\n" + "="*60)
    print("加载 LightGBM 模型")
    print("="*60)

    # 加载模型
    model = LightGBMClassifier()
    model.load_model(model_path)

    # 评估
    results = model.evaluate(X_test, y_test, dataset_name='Test')

    # 保存混淆矩阵
    cm_path = os.path.join(
        config.FIGURES_DIR,
        f'lightgbm_{dataset_tag}_eval_confusion_matrix.png'
    )
    plot_confusion_matrix(
        y_test, results['predictions'],
        config.CLASSES,
        save_path=cm_path,
        title=f'LightGBM Evaluation Confusion Matrix ({dataset_tag})'
    )

    # 保存分类报告
    report_path = os.path.join(
        config.LOGS_DIR,
        f'lightgbm_{dataset_tag}_eval_classification_report.txt'
    )
    print_classification_report(
        y_test, results['predictions'],
        config.CLASSES,
        save_path=report_path
    )

    return results

def evaluate_cnn(model_path, test_loader, y_test, dataset_tag):
    """
    评估CNN模型

    Args:
        model_path: 模型文件路径
        test_loader: 测试数据加载器
        y_test: 测试标签
        dataset_tag: 数据集标签

    Returns:
        预测结果和评估指标
    """
    print("\n" + "="*60)
    print("加载 CNN 模型")
    print("="*60)

    device = get_device()

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = CNN1D(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已加载: {model_path}")

    # 预测
    y_true, y_pred, y_prob = predict_cnn(model, test_loader, device)

    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    test_acc = accuracy_score(y_true, y_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    print(f"\n测试集准确率: {test_acc:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集召回率: {test_recall:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")

    # 保存混淆矩阵
    cm_path = os.path.join(
        config.FIGURES_DIR,
        f'cnn_{dataset_tag}_eval_confusion_matrix.png'
    )
    plot_confusion_matrix(
        y_true, y_pred,
        config.CLASSES,
        save_path=cm_path,
        title=f'CNN Evaluation Confusion Matrix ({dataset_tag})'
    )

    # 保存分类报告
    report_path = os.path.join(
        config.LOGS_DIR,
        f'cnn_{dataset_tag}_eval_classification_report.txt'
    )
    print_classification_report(
        y_true, y_pred,
        config.CLASSES,
        save_path=report_path
    )

    return {
        'predictions': y_pred,
        'probabilities': y_prob,
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }

def evaluate_lstm(model_path, test_loader, y_test, dataset_tag):
    """
    评估LSTM模型

    Args:
        model_path: 模型文件路径
        test_loader: 测试数据加载器
        y_test: 测试标签
        dataset_tag: 数据集标签

    Returns:
        预测结果和评估指标
    """
    print("\n" + "="*60)
    print("加载 LSTM 模型")
    print("="*60)

    device = get_device()

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = LSTMClassifier(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已加载: {model_path}")

    # 预测
    y_true, y_pred, y_prob = predict_lstm(model, test_loader, device)

    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    test_acc = accuracy_score(y_true, y_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    print(f"\n测试集准确率: {test_acc:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集召回率: {test_recall:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")

    # 保存混淆矩阵
    cm_path = os.path.join(
        config.FIGURES_DIR,
        f'lstm_{dataset_tag}_eval_confusion_matrix.png'
    )
    plot_confusion_matrix(
        y_true, y_pred,
        config.CLASSES,
        save_path=cm_path,
        title=f'LSTM Evaluation Confusion Matrix ({dataset_tag})'
    )

    # 保存分类报告
    report_path = os.path.join(
        config.LOGS_DIR,
        f'lstm_{dataset_tag}_eval_classification_report.txt'
    )
    print_classification_report(
        y_true, y_pred,
        config.CLASSES,
        save_path=report_path
    )

    return {
        'predictions': y_pred,
        'probabilities': y_prob,
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 确定数据集标签
    dataset_tag = args.dataset if args.mode == 'single' else 'mixed'

    # 打印配置信息
    print("\n" + "="*80)
    print("评估配置")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"数据模式: {args.mode}")
    if args.mode == 'single':
        print(f"数据集: {args.dataset}")
    else:
        print(f"数据集: {', '.join(config.DATASETS)}")
    print(f"归一化: {args.normalize}")
    print("="*80)

    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        # 使用默认路径
        if args.model == 'lightgbm':
            model_path = os.path.join(
                config.MODELS_DIR,
                f'lightgbm_{dataset_tag}_best.txt'
            )
        else:
            model_path = os.path.join(
                config.MODELS_DIR,
                f'{args.model}_{dataset_tag}_best.pth'
            )

    if not os.path.exists(model_path):
        print(f"\n错误: 模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        sys.exit(1)

    print(f"模型路径: {model_path}\n")

    # 准备数据
    normalize = args.normalize != 'none'
    data_dict = prepare_data(
        dataset_mode=args.mode,
        single_dataset=args.dataset,
        normalize=normalize,
        normalization_method=args.normalize if normalize else 'standard'
    )

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    # 评估模型
    if args.model == 'lightgbm':
        results = evaluate_lightgbm(model_path, X_test, y_test, dataset_tag)

    elif args.model in ['cnn', 'lstm']:
        # 创建DataLoader
        _, _, test_loader = create_dataloaders(
            data_dict['X_train'],
            data_dict['X_val'],
            X_test,
            data_dict['y_train'],
            data_dict['y_val'],
            y_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        if args.model == 'cnn':
            results = evaluate_cnn(model_path, test_loader, y_test, dataset_tag)
        else:
            results = evaluate_lstm(model_path, test_loader, y_test, dataset_tag)

    # 保存预测结果
    if args.save_predictions:
        pred_path = os.path.join(
            config.LOGS_DIR,
            f'{args.model}_{dataset_tag}_predictions.npz'
        )
        np.savez(
            pred_path,
            y_true=y_test,
            y_pred=results['predictions'],
            y_prob=results['probabilities']
        )
        print(f"\n预测结果已保存至: {pred_path}")

    print("\n评估完成!")
    print(f"结果已保存至:")
    print(f"  图表: {config.FIGURES_DIR}")
    print(f"  日志: {config.LOGS_DIR}\n")

if __name__ == '__main__':
    main()
