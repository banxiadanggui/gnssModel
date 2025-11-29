"""
6分类CNN模型
区分正常、跟踪失败和4种数据集来源的干扰
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config_6class as config
from utils import (plot_confusion_matrix, print_classification_report,
                  plot_training_history, save_results, EarlyStopping,
                  get_device, print_model_summary)

class CNN1D_6Class(nn.Module):
    """6分类1D卷积神经网络"""

    def __init__(self, input_channels=9, num_classes=6, dropout=0.5):
        """
        Args:
            input_channels: 输入特征数 (9)
            num_classes: 类别数 (6)
            dropout: Dropout概率
        """
        super(CNN1D_6Class, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 第四个卷积块（增加一层以提高6分类能力）
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: 输入数据 shape (batch, time, features) -> (B, 2000, 9)

        Returns:
            输出logits shape (batch, num_classes)
        """
        # 转置: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)  # (B, 9, 2000)

        # 卷积层
        x = self.conv1(x)  # (B, 64, 1000)
        x = self.conv2(x)  # (B, 128, 500)
        x = self.conv3(x)  # (B, 256, 250)
        x = self.conv4(x)  # (B, 512, 1)

        # 展平
        x = x.squeeze(-1)  # (B, 512)

        # 全连接层
        x = self.fc(x)  # (B, num_classes)

        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc

def predict(model, dataloader, device):
    """预测"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def train_cnn_6class(train_loader, val_loader, test_loader):
    """
    训练6分类CNN模型的完整流程

    Args:
        train_loader, val_loader, test_loader: DataLoader

    Returns:
        模型和评估结果
    """
    print("\n" + "="*80)
    print("开始训练 6分类CNN 模型")
    print("="*80)
    print("分类目标:")
    for idx, class_name in config.IDX_TO_CLASS_6.items():
        print(f"  {idx}: {class_name}")
    print("="*80 + "\n")

    # 设置设备
    device = get_device()

    # 创建模型
    model = CNN1D_6Class(**config.CNN_6CLASS_CONFIG).to(device)
    print_model_summary(model)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.CNN_6CLASS_TRAIN_CONFIG['learning_rate'],
        weight_decay=config.CNN_6CLASS_TRAIN_CONFIG['weight_decay']
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.CNN_6CLASS_TRAIN_CONFIG['scheduler_patience'],
        factor=config.CNN_6CLASS_TRAIN_CONFIG['scheduler_factor']
    )

    # 早停机制
    early_stopping = EarlyStopping(
        patience=config.CNN_6CLASS_TRAIN_CONFIG['early_stopping_patience'],
        mode='min'
    )

    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    best_model_state = None

    # 训练循环
    num_epochs = config.CNN_6CLASS_TRAIN_CONFIG['epochs']
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印进度
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # 早停检查
        if early_stopping(val_loss):
            print(f"\n早停触发于 epoch {epoch+1}")
            break

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    print(f"\n训练完成! 最佳验证Loss: {best_val_loss:.4f}")
    print("="*80 + "\n")

    # 评估
    print("评估模型...")
    y_train, pred_train, _ = predict(model, train_loader, device)
    y_val, pred_val, _ = predict(model, val_loader, device)
    y_test, pred_test, prob_test = predict(model, test_loader, device)

    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    train_acc = accuracy_score(y_train, pred_train)
    val_acc = accuracy_score(y_val, pred_val)
    test_acc = accuracy_score(y_test, pred_test)

    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, pred_test, average='weighted'
    )

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 保存结果
    dataset_tag = '6class'

    # 保存训练曲线
    history_path = os.path.join(
        config.FIGURES_DIR,
        f'cnn_{dataset_tag}_training_history.png'
    )
    plot_training_history(history, save_path=history_path)

    # 保存混淆矩阵
    cm_path = os.path.join(
        config.FIGURES_DIR,
        f'cnn_{dataset_tag}_confusion_matrix.png'
    )
    plot_confusion_matrix(
        y_test, pred_test,
        config.CLASSES_6,
        save_path=cm_path,
        title=f'CNN 6-Class Confusion Matrix'
    )

    # 打印分类报告
    report_path = os.path.join(
        config.LOGS_DIR,
        f'cnn_{dataset_tag}_classification_report.txt'
    )
    print_classification_report(
        y_test, pred_test,
        config.CLASSES_6,
        save_path=report_path
    )

    # 保存模型
    model_path = os.path.join(
        config.MODELS_DIR,
        f'cnn_{dataset_tag}_best.pth'
    )
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': config.CNN_6CLASS_CONFIG,
        'history': history,
        'classes': config.CLASSES_6
    }, model_path)
    print(f"模型已保存至: {model_path}")

    # 保存结果
    results = {
        'model_name': 'CNN_6Class',
        'dataset_mode': '6class',
        'num_classes': config.NUM_CLASSES_6,
        'classes': config.CLASSES_6,
        'train_metrics': {'accuracy': float(train_acc)},
        'val_metrics': {'accuracy': float(val_acc)},
        'test_metrics': {
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1)
        },
        'best_val_loss': float(best_val_loss),
        'total_epochs': len(history['train_loss'])
    }

    save_results(results, config.LOGS_DIR, 'cnn', dataset_tag)

    return model, results
