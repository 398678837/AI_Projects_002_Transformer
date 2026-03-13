"""
支持向量机（SVM）算法的scikit-learn实现
使用鸢尾花数据集进行多分类
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()

X = iris.data  # 所有特征
y = iris.target  # 所有标签

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")

# 2. 数据预处理 - 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 4. 创建并训练模型
print("\n训练SVM模型...")
# kernel: 'linear', 'poly', 'rbf', 'sigmoid'
# C: 正则化参数
# gamma: 核函数参数
model = SVC(
    kernel='rbf',  # 径向基函数核
    C=1.0,         # 正则化参数
    gamma='scale',  # 核函数参数
    random_state=42
)
model.fit(X_train, y_train)

# 5. 模型预测
print("\n预测测试集...")
y_pred = model.predict(X_test)

# 6. 模型评估
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n模型准确率: {accuracy:.2f}")
print("混淆矩阵:")
print(conf_matrix)

# 7. 查看模型参数
print("\n模型参数:")
print(f" 核函数: {model.kernel}")
print(f" 正则化参数C: {model.C}")
print(f" 核函数参数gamma: {model.gamma}")

# 8. 简单的预测示例
print("\n预测示例:")
# 随机选择一个测试样本
sample_idx = 0
sample = X_test[sample_idx]
actual_label = y_test[sample_idx]
predicted_label = y_pred[sample_idx]

print(f"样本特征: {sample}")
print(f"实际类别: {iris.target_names[actual_label]}")
print(f"预测类别: {iris.target_names[predicted_label]}")

# 9. 测试不同核函数的效果
print("\n测试不同核函数的效果:")
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    model_kernel = SVC(
        kernel=kernel,
        C=1.0,
        gamma='scale',
        random_state=42
    )
    model_kernel.fit(X_train, y_train)
    y_pred_kernel = model_kernel.predict(X_test)
    acc_kernel = accuracy_score(y_test, y_pred_kernel)
    print(f"kernel={kernel}: 准确率={acc_kernel:.2f}")

# 10. 测试不同C值的效果
print("\n测试不同C值的效果:")
C_values = [0.1, 1, 10, 100]
for C in C_values:
    model_C = SVC(
        kernel='rbf',
        C=C,
        gamma='scale',
        random_state=42
    )
    model_C.fit(X_train, y_train)
    y_pred_C = model_C.predict(X_test)
    acc_C = accuracy_score(y_test, y_pred_C)
    print(f"C={C}: 准确率={acc_C:.2f}")

# 11. 可视化决策边界（仅使用前两个特征）
print("\n可视化决策边界...")
# 仅使用前两个特征进行可视化
X_2d = X_scaled[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

# 训练2D模型
model_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model_2d.fit(X_train_2d, y_train_2d)

# 创建网格
h = 0.02  # 网格步长
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测网格点
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d, edgecolors='k', cmap=plt.cm.coolwarm)
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d, edgecolors='k', marker='s', cmap=plt.cm.coolwarm)
plt.xlabel('sepal length (标准化)')
plt.ylabel('sepal width (标准化)')
plt.title('SVM决策边界 (RBF核)')
plt.savefig('svm_decision_boundary.png')
print("决策边界可视化已保存为 'svm_decision_boundary.png'")

print("\nscikit-learn SVM Demo完成！")
