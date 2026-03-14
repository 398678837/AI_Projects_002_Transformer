"""
KNN（K最近邻）算法的scikit-learn实现
使用鸢尾花数据集进行多分类
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()

X = iris.data  # 所有特征
y = iris.target  # 所有标签

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"Class名称: {iris.target_names}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 3. 创建并训练模型
print("\n训练KNN模型...")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. 模型预测
print("\n预测测试集...")
y_pred = model.predict(X_test)

# 5. 模型评估
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n模型Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# 6. 查看模型参数
print("\n模型参数:")
print(f"K值: {model.n_neighbors}")
print(f"距离度量: {model.metric}")

# 7. 简单的预测示例
print("\n预测示例:")
# 随机选择一个测试样本
sample_idx = 0
sample = X_test[sample_idx]
actual_label = y_test[sample_idx]
predicted_label = y_pred[sample_idx]

print(f"样本特征: {sample}")
print(f"实际Class: {iris.target_names[actual_label]}")
print(f"预测Class: {iris.target_names[predicted_label]}")

# 8. 测试不同K值的效果
print("\n测试不同K值的效果:")
for k in [1, 3, 5, 7, 9]:
    model_k = KNeighborsClassifier(n_neighbors=k)
    model_k.fit(X_train, y_train)
    y_pred_k = model_k.predict(X_test)
    acc_k = accuracy_score(y_test, y_pred_k)
    print(f"K={k}: Accuracy={acc_k:.2f}")

print("\nscikit-learn KNN Demo完成！")
