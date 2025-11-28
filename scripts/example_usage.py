"""
使用示例脚本
展示如何通过Python代码直接使用训练管线
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import prepare_data, create_dataloaders
from models.lightgbm_model import train_lightgbm
from models.cnn_model import train_cnn
from models.lstm_model import train_lstm
from utils import set_seed

def example_train_single_dataset():
    """示例1: 在单个数据集上训练LightGBM"""
    print("\n" + "="*80)
    print("示例1: 在UTD数据集上训练LightGBM")
    print("="*80)

    # 设置随机种子
    set_seed(42)

    # 准备数据
    data_dict = prepare_data(
        dataset_mode='single',
        single_dataset='UTD',
        normalize=True,
        normalization_method='standard'
    )

    # 训练LightGBM
    model, results = train_lightgbm(
        data_dict,
        dataset_mode='single',
        single_dataset='UTD'
    )

    print(f"\n训练完成!")
    print(f"测试集准确率: {results['test_metrics']['accuracy']:.4f}")
    print(f"测试集F1分数: {results['test_metrics']['f1']:.4f}")

def example_train_mixed_dataset():
    """示例2: 在混合数据集上训练CNN"""
    print("\n" + "="*80)
    print("示例2: 在混合数据集上训练CNN")
    print("="*80)

    # 设置随机种子
    set_seed(42)

    # 准备数据
    data_dict = prepare_data(
        dataset_mode='mixed',
        normalize=True,
        normalization_method='standard'
    )

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict['X_train'],
        data_dict['X_val'],
        data_dict['X_test'],
        data_dict['y_train'],
        data_dict['y_val'],
        data_dict['y_test'],
        batch_size=64,
        num_workers=0
    )

    # 训练CNN
    model, results = train_cnn(
        train_loader, val_loader, test_loader,
        dataset_mode='mixed',
        single_dataset='UTD'
    )

    print(f"\n训练完成!")
    print(f"测试集准确率: {results['test_metrics']['accuracy']:.4f}")
    print(f"测试集F1分数: {results['test_metrics']['f1']:.4f}")

def example_train_all_models():
    """示例3: 训练所有模型并对比结果"""
    print("\n" + "="*80)
    print("示例3: 在混合数据集上训练所有模型并对比")
    print("="*80)

    # 设置随机种子
    set_seed(42)

    # 准备数据（只需要准备一次）
    data_dict = prepare_data(
        dataset_mode='mixed',
        normalize=True,
        normalization_method='standard'
    )

    # 创建DataLoader（用于CNN和LSTM）
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict['X_train'],
        data_dict['X_val'],
        data_dict['X_test'],
        data_dict['y_train'],
        data_dict['y_val'],
        data_dict['y_test'],
        batch_size=64,
        num_workers=0
    )

    # 存储所有结果
    all_results = {}

    # 训练LightGBM
    print("\n" + "-"*80)
    print("训练 LightGBM...")
    print("-"*80)
    _, results = train_lightgbm(
        data_dict,
        dataset_mode='mixed',
        single_dataset='UTD'
    )
    all_results['LightGBM'] = results['test_metrics']

    # 训练CNN
    print("\n" + "-"*80)
    print("训练 CNN...")
    print("-"*80)
    _, results = train_cnn(
        train_loader, val_loader, test_loader,
        dataset_mode='mixed',
        single_dataset='UTD'
    )
    all_results['CNN'] = results['test_metrics']

    # 训练LSTM
    print("\n" + "-"*80)
    print("训练 LSTM...")
    print("-"*80)
    _, results = train_lstm(
        train_loader, val_loader, test_loader,
        dataset_mode='mixed',
        single_dataset='UTD'
    )
    all_results['LSTM'] = results['test_metrics']

    # 打印对比结果
    print("\n" + "="*80)
    print("所有模型测试集结果对比")
    print("="*80)
    print(f"{'模型':<15} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12}")
    print("-"*80)
    for model_name, metrics in all_results.items():
        print(f"{model_name:<15} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f}")
    print("="*80 + "\n")

def example_load_and_predict():
    """示例4: 加载已保存的模型并进行预测"""
    print("\n" + "="*80)
    print("示例4: 加载已保存的模型并预测")
    print("="*80)

    import torch
    import numpy as np
    import config
    from models.cnn_model import CNN1D, predict
    from utils import get_device

    # 准备数据
    data_dict = prepare_data(
        dataset_mode='mixed',
        normalize=True,
        normalization_method='standard'
    )

    # 创建测试数据加载器
    _, _, test_loader = create_dataloaders(
        data_dict['X_train'],
        data_dict['X_val'],
        data_dict['X_test'],
        data_dict['y_train'],
        data_dict['y_val'],
        data_dict['y_test'],
        batch_size=64,
        num_workers=0
    )

    # 模型路径
    model_path = os.path.join(config.MODELS_DIR, 'cnn_mixed_best.pth')

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型")
        return

    # 加载模型
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    model = CNN1D(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已加载: {model_path}")

    # 预测
    y_true, y_pred, y_prob = predict(model, test_loader, device)

    # 计算准确率
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"预测样本数: {len(y_pred)}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=config.CLASSES))

    # 显示一些预测示例
    print("\n前10个样本的预测结果:")
    print(f"{'真实标签':<15} {'预测标签':<15} {'预测概率':<30}")
    print("-"*60)
    for i in range(min(10, len(y_pred))):
        true_label = config.IDX_TO_CLASS[y_true[i]]
        pred_label = config.IDX_TO_CLASS[y_pred[i]]
        probs = ', '.join([f'{p:.3f}' for p in y_prob[i]])
        print(f"{true_label:<15} {pred_label:<15} [{probs}]")

def example_custom_training():
    """示例5: 自定义训练参数"""
    print("\n" + "="*80)
    print("示例5: 使用自定义参数训练LSTM")
    print("="*80)

    import torch
    import config
    from models.lstm_model import LSTMClassifier, train_epoch, validate_epoch
    from utils import get_device, EarlyStopping
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # 准备数据
    data_dict = prepare_data(
        dataset_mode='single',
        single_dataset='UTD',
        normalize=True
    )

    # 创建DataLoader，使用自定义批次大小
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict['X_train'],
        data_dict['X_val'],
        data_dict['X_test'],
        data_dict['y_train'],
        data_dict['y_val'],
        data_dict['y_test'],
        batch_size=32,  # 自定义批次大小
        num_workers=0
    )

    # 设置设备
    device = get_device()

    # 创建模型，使用自定义参数
    model = LSTMClassifier(
        input_size=9,
        hidden_size=64,  # 自定义隐藏层大小
        num_layers=3,    # 自定义层数
        num_classes=3,
        dropout=0.4,     # 自定义dropout
        bidirectional=True
    ).to(device)

    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 自定义优化器和学习率
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,  # 自定义学习率
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    # 早停
    early_stopping = EarlyStopping(patience=10, mode='min')

    # 训练循环
    num_epochs = 20  # 自定义训练轮数
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if early_stopping(val_loss):
            print(f"早停触发于 epoch {epoch+1}")
            break

    print("\n自定义训练完成!")

if __name__ == '__main__':
    """
    运行示例：
    python scripts\example_usage.py

    可以取消注释下面的函数来运行不同的示例
    """

    print("\n" + "="*80)
    print("GNSS干扰检测训练管线使用示例")
    print("="*80)
    print("\n可用示例:")
    print("1. example_train_single_dataset() - 在单个数据集上训练LightGBM")
    print("2. example_train_mixed_dataset() - 在混合数据集上训练CNN")
    print("3. example_train_all_models() - 训练所有模型并对比")
    print("4. example_load_and_predict() - 加载模型并预测")
    print("5. example_custom_training() - 自定义训练参数")
    print("\n取消注释下面的代码来运行示例")
    print("="*80 + "\n")

    # 取消注释来运行示例
    # example_train_single_dataset()
    # example_train_mixed_dataset()
    # example_train_all_models()
    # example_load_and_predict()
    # example_custom_training()
