"""
6分类CNN模型训练脚本
运行: python scripts/train_cnn_6class.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_6class import prepare_6class_data, create_6class_dataloaders
from models.cnn_6class_model import train_cnn_6class
from utils import set_seed
import config_6class as config

def main():
    print("\n" + "="*80)
    print("6分类GNSS干扰检测 - CNN模型训练")
    print("="*80)
    print("任务说明:")
    print("  将GNSS信号分为6类:")
    print("    0: normal         - 正常信号")
    print("    1: tracking_fail  - 跟踪失败")
    print("    2: attack_UTD     - UTD数据集干扰")
    print("    3: attack_TGS     - TGS数据集干扰")
    print("    4: attack_TGD     - TGD数据集干扰")
    print("    5: attack_MCD     - MCD数据集干扰")
    print("="*80 + "\n")

    # 设置随机种子
    set_seed(config.RANDOM_SEED)

    # 准备数据
    print("步骤1: 准备数据")
    print("-" * 80)
    data_dict = prepare_6class_data(
        normalize=True,
        normalization_method=config.NORMALIZATION
    )

    # 创建DataLoader
    print("步骤2: 创建DataLoader")
    print("-" * 80)
    train_loader, val_loader, test_loader = create_6class_dataloaders(
        data_dict['X_train'],
        data_dict['X_val'],
        data_dict['X_test'],
        data_dict['y_train'],
        data_dict['y_val'],
        data_dict['y_test'],
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # 训练模型
    print("步骤3: 训练6分类CNN模型")
    print("-" * 80)
    model, results = train_cnn_6class(train_loader, val_loader, test_loader)

    # 打印最终结果
    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    print(f"测试集准确率: {results['test_metrics']['accuracy']:.4f}")
    print(f"测试集精确率: {results['test_metrics']['precision']:.4f}")
    print(f"测试集召回率: {results['test_metrics']['recall']:.4f}")
    print(f"测试集F1分数: {results['test_metrics']['f1']:.4f}")
    print("="*80)

    print("\n结果文件:")
    print(f"  模型文件:   results/models/cnn_6class_best.pth")
    print(f"  训练曲线:   results/figures/cnn_6class_training_history.png")
    print(f"  混淆矩阵:   results/figures/cnn_6class_confusion_matrix.png")
    print(f"  分类报告:   results/logs/cnn_6class_classification_report.txt")
    print(f"  结果JSON:   results/logs/cnn_6class_*_results.json")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
