"""
LSTM模型
适合处理时间序列依赖关系
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from utils import (plot_confusion_matrix, print_classification_report,
                  plot_training_history, save_results, EarlyStopping,
                  get_device, print_model_summary)

class LSTMClassifier(nn.Module):
    """LSTM分类器"""

    def __init__(self, input_size=9, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.3, bidirectional=True):
        """
        Args:
            input_size: 输入特征数 (9)
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            num_classes: 类别数 (3)
            dropout: Dropout概率
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_directions, 128),
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
        # LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, T, H*num_directions)

        # 取最后一个时间步的输出
        # 如果是双向，h_n的shape是(num_layers*2, B, H)
        if self.bidirectional:
            # 取最后一层的前向和后向hidden state
            h_forward = h_n[-2, :, :]  # (B, H)
            h_backward = h_n[-1, :, :]  # (B, H)
            h_last = torch.cat([h_forward, h_backward], dim=1)  # (B, H*2)
        else:
            h_last = h_n[-1, :, :]  # (B, H)

        # 全连接层
        out = self.fc(h_last)  # (B, num_classes)

        return out

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
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

def train_lstm(train_loader, val_loader, test_loader,
               dataset_mode='mixed', single_dataset='UTD'):
    """
    训练LSTM模型的完整流程

    Args:
        train_loader, val_loader, test_loader: DataLoader
        dataset_mode: 'single' 或 'mixed'
        single_dataset: 如果是single模式，数据集名称

    Returns:
        模型和评估结果
    """
    print("\n" + "="*60)
    print("开始训练 LSTM 模型")
    print("="*60)

    # 设置设备
    device = get_device()

    # 创建模型
    model = LSTMClassifier(**config.LSTM_CONFIG).to(device)
    print_model_summary(model)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LSTM_TRAIN_CONFIG['learning_rate'],
        weight_decay=config.LSTM_TRAIN_CONFIG['weight_decay']
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.LSTM_TRAIN_CONFIG['scheduler_patience'],
        factor=config.LSTM_TRAIN_CONFIG['scheduler_factor'],
        verbose=True
    )

    # 早停机制
    early_stopping = EarlyStopping(
        patience=config.LSTM_TRAIN_CONFIG['early_stopping_patience'],
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
    num_epochs = config.LSTM_TRAIN_CONFIG['epochs']
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
    print("="*60 + "\n")

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
    dataset_tag = single_dataset if dataset_mode == 'single' else 'mixed'

    # 保存训练曲线
    history_path = os.path.join(
        config.FIGURES_DIR,
        f'lstm_{dataset_tag}_training_history.png'
    )
    plot_training_history(history, save_path=history_path)

    # 保存混淆矩阵
    cm_path = os.path.join(
        config.FIGURES_DIR,
        f'lstm_{dataset_tag}_confusion_matrix.png'
    )
    plot_confusion_matrix(
        y_test, pred_test,
        config.CLASSES,
        save_path=cm_path,
        title=f'LSTM Confusion Matrix ({dataset_tag})'
    )

    # 打印分类报告
    report_path = os.path.join(
        config.LOGS_DIR,
        f'lstm_{dataset_tag}_classification_report.txt'
    )
    print_classification_report(
        y_test, pred_test,
        config.CLASSES,
        save_path=report_path
    )

    # 保存模型
    model_path = os.path.join(
        config.MODELS_DIR,
        f'lstm_{dataset_tag}_best.pth'
    )
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': config.LSTM_CONFIG,
        'history': history
    }, model_path)
    print(f"模型已保存至: {model_path}")

    # 保存结果
    results = {
        'model_name': 'LSTM',
        'dataset_mode': dataset_mode,
        'dataset': single_dataset if dataset_mode == 'single' else 'mixed',
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

    save_results(results, config.LOGS_DIR, 'lstm', dataset_tag)

    return model, results
