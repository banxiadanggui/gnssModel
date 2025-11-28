"""
LightGBM模型
适合快速baseline和特征重要性分析
"""
import numpy as np
import lightgbm as lgb
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from utils import plot_feature_importance, plot_confusion_matrix
from utils import print_classification_report, save_results

class LightGBMClassifier:
    """LightGBM分类器封装"""

    def __init__(self, params=None):
        """
        Args:
            params: LightGBM参数字典
        """
        self.params = params if params is not None else config.LIGHTGBM_PARAMS
        self.model = None
        self.feature_importance = None

    def train(self, X_train, y_train, X_val, y_val,
              num_boost_round=500, early_stopping_rounds=50, verbose_eval=50):
        """
        训练模型

        Args:
            X_train: 训练数据 shape (N, 2000, 9)
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            num_boost_round: 最大迭代轮数
            early_stopping_rounds: 早停轮数
            verbose_eval: 打印频率
        """
        print("\n" + "="*60)
        print("开始训练 LightGBM 模型")
        print("="*60)

        # 将3D数据展平为2D: (N, 2000, 9) -> (N, 2000*9)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        print(f"训练集形状: {X_train_flat.shape}")
        print(f"验证集形状: {X_val_flat.shape}")

        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train_flat, label=y_train)
        val_data = lgb.Dataset(X_val_flat, label=y_val, reference=train_data)

        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=verbose_eval)
            ]
        )

        # 获取特征重要性
        self.feature_importance = self.model.feature_importance(importance_type='gain')

        print(f"\n最佳迭代: {self.model.best_iteration}")
        print("="*60 + "\n")

        return self.model

    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        X_flat = X.reshape(X.shape[0], -1)
        pred_proba = self.model.predict(X_flat, num_iteration=self.model.best_iteration)
        pred = np.argmax(pred_proba, axis=1)
        return pred

    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        X_flat = X.reshape(X.shape[0], -1)
        pred_proba = self.model.predict(X_flat, num_iteration=self.model.best_iteration)
        return pred_proba

    def evaluate(self, X, y, dataset_name='Test'):
        """评估模型"""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted'
        )

        print(f"\n{dataset_name} 集评估结果:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def plot_feature_importance(self, top_n=20, save_path=None):
        """绘制特征重要性"""
        if self.feature_importance is None:
            raise ValueError("需要先训练模型")

        # 生成特征名称 (feature_0_0, feature_0_1, ..., feature_1999_8)
        feature_names = []
        for t in range(config.WINDOW_SIZE):
            for f in range(config.NUM_FEATURES):
                feature_names.append(f"t{t}_f{f}")

        plot_feature_importance(
            feature_names,
            self.feature_importance,
            top_n=top_n,
            save_path=save_path
        )

    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        self.model.save_model(filepath)
        print(f"模型已保存至: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        self.model = lgb.Booster(model_file=filepath)
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        print(f"模型已加载: {filepath}")

def train_lightgbm(data_dict, dataset_mode='mixed', single_dataset='UTD'):
    """
    训练LightGBM模型的完整流程

    Args:
        data_dict: 包含数据的字典
        dataset_mode: 'single' 或 'mixed'
        single_dataset: 如果是single模式，数据集名称

    Returns:
        模型和评估结果
    """
    # 提取数据
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']

    # 创建模型
    model = LightGBMClassifier()

    # 训练
    model.train(
        X_train, y_train, X_val, y_val,
        **config.LIGHTGBM_TRAIN_PARAMS
    )

    # 评估
    train_results = model.evaluate(X_train, y_train, 'Train')
    val_results = model.evaluate(X_val, y_val, 'Validation')
    test_results = model.evaluate(X_test, y_test, 'Test')

    # 保存混淆矩阵
    dataset_tag = single_dataset if dataset_mode == 'single' else 'mixed'
    cm_path = os.path.join(
        config.FIGURES_DIR,
        f'lightgbm_{dataset_tag}_confusion_matrix.png'
    )
    plot_confusion_matrix(
        y_test, test_results['predictions'],
        config.CLASSES,
        save_path=cm_path,
        title=f'LightGBM Confusion Matrix ({dataset_tag})'
    )

    # 保存特征重要性
    fi_path = os.path.join(
        config.FIGURES_DIR,
        f'lightgbm_{dataset_tag}_feature_importance.png'
    )
    model.plot_feature_importance(top_n=20, save_path=fi_path)

    # 打印分类报告
    report_path = os.path.join(
        config.LOGS_DIR,
        f'lightgbm_{dataset_tag}_classification_report.txt'
    )
    metrics = print_classification_report(
        y_test, test_results['predictions'],
        config.CLASSES,
        save_path=report_path
    )

    # 保存模型
    model_path = os.path.join(
        config.MODELS_DIR,
        f'lightgbm_{dataset_tag}_best.txt'
    )
    model.save_model(model_path)

    # 保存结果
    results = {
        'model_name': 'LightGBM',
        'dataset_mode': dataset_mode,
        'dataset': single_dataset if dataset_mode == 'single' else 'mixed',
        'train_metrics': {
            'accuracy': float(train_results['accuracy']),
            'precision': float(train_results['precision']),
            'recall': float(train_results['recall']),
            'f1': float(train_results['f1'])
        },
        'val_metrics': {
            'accuracy': float(val_results['accuracy']),
            'precision': float(val_results['precision']),
            'recall': float(val_results['recall']),
            'f1': float(val_results['f1'])
        },
        'test_metrics': {
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1': float(test_results['f1'])
        },
        'best_iteration': int(model.model.best_iteration)
    }

    save_results(results, config.LOGS_DIR, 'lightgbm', dataset_tag)

    return model, results
