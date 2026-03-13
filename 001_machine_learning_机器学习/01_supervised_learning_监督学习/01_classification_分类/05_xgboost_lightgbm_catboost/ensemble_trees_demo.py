"""
集成树模型（XGBoost, LightGBM, CatBoost）的实现
使用鸢尾花数据集进行多分类
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 尝试导入XGBoost, LightGBM, CatBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost未安装，将跳过XGBoost演示")
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("LightGBM未安装，将跳过LightGBM演示")
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("CatBoost未安装，将跳过CatBoost演示")
    CatBoostClassifier = None

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()

X = iris.data  # 所有特征
y = iris.target  # 所有标签

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 3. 模型训练和评估函数
def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    """训练模型并评估性能"""
    print(f"\n训练{model_name}模型...")
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name}准确率: {accuracy:.2f}")
    print("混淆矩阵:")
    print(conf_matrix)
    
    # 特征重要性（如果支持）
    if hasattr(model, 'feature_importances_'):
        print("\n特征重要性:")
        feature_importance = model.feature_importances_
        for i, (feature, importance) in enumerate(zip(iris.feature_names, feature_importance)):
            print(f"{feature}: {importance:.4f}")
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        plt.barh(iris.feature_names, feature_importance, color='skyblue')
        plt.title(f'{model_name}特征重要性')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
        print(f"特征重要性可视化已保存为 '{model_name.lower().replace(' ', '_')}_feature_importance.png'")
    
    return accuracy

# 4. 训练和评估各个模型
accuracies = {}

# XGBoost
if XGBClassifier:
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    accuracies['XGBoost'] = train_and_evaluate('XGBoost', xgb_model, X_train, y_train, X_test, y_test)

# LightGBM
if LGBMClassifier:
    lgb_model = LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    accuracies['LightGBM'] = train_and_evaluate('LightGBM', lgb_model, X_train, y_train, X_test, y_test)

# CatBoost
if CatBoostClassifier:
    cat_model = CatBoostClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=0  # 关闭详细输出
    )
    accuracies['CatBoost'] = train_and_evaluate('CatBoost', cat_model, X_train, y_train, X_test, y_test)

# 5. 模型性能对比
if accuracies:
    print("\n模型性能对比:")
    for model_name, acc in accuracies.items():
        print(f"{model_name}: {acc:.2f}")
    
    # 可视化性能对比
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color='lightgreen')
    plt.title('集成树模型性能对比')
    plt.ylabel('准确率')
    plt.ylim(0.8, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('ensemble_models_comparison.png')
    print("模型性能对比可视化已保存为 'ensemble_models_comparison.png'")

# 6. 简单的预测示例
print("\n预测示例:")
# 定义新样本
new_samples = [
    [5.1, 3.5, 1.4, 0.2],  # 可能是setosa
    [6.2, 2.9, 4.3, 1.3],  # 可能是versicolor
    [7.3, 2.9, 6.3, 1.8]   # 可能是virginica
]

# 使用第一个可用的模型进行预测
if XGBClassifier:
    predictions = xgb_model.predict(new_samples)
    print("XGBoost预测结果:")
    for i, (sample, pred) in enumerate(zip(new_samples, predictions)):
        print(f"样本 {i+1}: {sample} → 预测类别: {iris.target_names[pred]}")
elif LGBMClassifier:
    predictions = lgb_model.predict(new_samples)
    print("LightGBM预测结果:")
    for i, (sample, pred) in enumerate(zip(new_samples, predictions)):
        print(f"样本 {i+1}: {sample} → 预测类别: {iris.target_names[pred]}")
elif CatBoostClassifier:
    predictions = cat_model.predict(new_samples)
    print("CatBoost预测结果:")
    for i, (sample, pred) in enumerate(zip(new_samples, predictions)):
        print(f"样本 {i+1}: {sample} → 预测类别: {iris.target_names[pred]}")

print("\n集成树模型Demo完成！")
