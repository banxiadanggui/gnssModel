"""
工具函数模块
包含训练、评估、可视化等通用工具
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import random
import os
import json
from datetime import datetime

def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None, title='Confusion Matrix'):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
        save_path: 保存路径
        title: 图表标题
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 添加每行的准确率
    for i in range(len(classes)):
        row_sum = cm[i].sum()
        if row_sum > 0:
            accuracy = cm[i, i] / row_sum * 100
            plt.text(len(classes) + 0.5, i + 0.5, f'{accuracy:.1f}%',
                    ha='center', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线

    Args:
        history: 包含loss和accuracy的字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制Loss曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制Accuracy曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def print_classification_report(y_true, y_pred, classes, save_path=None):
    """
    打印并保存分类报告

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
        save_path: 保存路径
    """
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(report)
    print("="*60)

    # 计算总体指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    print(f"\n整体指标:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("="*60 + "\n")

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Classification Report:\n")
            f.write("="*60 + "\n")
            f.write(report + "\n")
            f.write("="*60 + "\n")
            f.write(f"\n整体指标:\n")
            f.write(f"  Accuracy:  {accuracy:.4f}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1-Score:  {f1:.4f}\n")
            f.write("="*60 + "\n")
        print(f"分类报告已保存至: {save_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_feature_importance(feature_names, importance, top_n=20, save_path=None):
    """
    绘制特征重要性（用于LightGBM）

    Args:
        feature_names: 特征名称列表
        importance: 特征重要性值
        top_n: 显示前N个重要特征
        save_path: 保存路径
    """
    # 按重要性排序
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def save_results(results, save_dir, model_name, dataset_mode):
    """
    保存实验结果

    Args:
        results: 结果字典
        save_dir: 保存目录
        model_name: 模型名称
        dataset_mode: 数据集模式 ('single' 或 'mixed')
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{dataset_mode}_{timestamp}_results.json"
    filepath = os.path.join(save_dir, filename)

    # 将numpy类型转换为Python原生类型
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    results_converted = convert_types(results)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=4, ensure_ascii=False)

    print(f"结果已保存至: {filepath}")
    return filepath

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善值
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

def get_device():
    """获取可用的计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """打印模型摘要"""
    print("\n" + "="*60)
    print("Model Summary:")
    print("="*60)
    print(model)
    print("="*60)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print("="*60 + "\n")
