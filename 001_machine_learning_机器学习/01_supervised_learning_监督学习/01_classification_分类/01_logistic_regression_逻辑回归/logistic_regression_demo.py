"""
逻辑回归最简单的Demo
使用scikit-learn库和鸢尾花数据集实现二分类
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()

# 只取前两个类别（Setosa和Versicolor）进行二分类
X = iris.data[:100]  # 前100个样本
y = iris.target[:100]  # 前100个标签（0和1）

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names[:2]}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 3. 创建并训练逻辑回归模型
print("\n训练逻辑回归模型...")
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 模型预测
print("\n预测测试集...")
y_pred = model.predict(X_test)

# 5. 模型评估
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n模型准确率: {accuracy:.2f}")
print("混淆矩阵:")
print(conf_matrix)

# 6. 查看模型参数
print("\n模型参数:")
print(f"系数: {model.coef_}")
print(f"截距: {model.intercept_}")

# 7. 简单的预测示例
print("\n预测示例:")
# 随机选择一个测试样本
sample_idx = 0
sample = X_test[sample_idx]
actual_label = y_test[sample_idx]
predicted_label = y_pred[sample_idx]

print(f"样本特征: {sample}")
print(f"实际类别: {iris.target_names[actual_label]}")
print(f"预测类别: {iris.target_names[predicted_label]}")

print("\n逻辑回归Demo完成！")
