"""
决策树（Decision Tree）算法的scikit-learn实现
使用鸢尾花数据集进行多分类
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

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
print("\n训练决策树模型...")
# criterion: 'gini' for Gini impurity, 'entropy' for information gain
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
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
print(f" criterion: {model.criterion}")
print(f" max_depth: {model.max_depth}")
print(f" Feature Importance: {model.feature_importances_}")

# 7. 可视化决策树
print("\n可视化决策树...")
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Decision Tree Visualization')
plt.savefig('images/decision_tree_visualization.png')
print("决策树可视化已保存为 'decision_tree_visualization.png'")

# 8. 简单的预测示例
print("\n预测示例:")
# 随机选择一个测试样本
sample_idx = 0
sample = X_test[sample_idx]
actual_label = y_test[sample_idx]
predicted_label = y_pred[sample_idx]

print(f"样本特征: {sample}")
print(f"实际Class: {iris.target_names[actual_label]}")
print(f"预测Class: {iris.target_names[predicted_label]}")

# 9. 测试不同max_depth的效果
print("\n测试不同max_depth的效果:")
for depth in [1, 2, 3, 4, 5, None]:
    model_depth = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
    model_depth.fit(X_train, y_train)
    y_pred_depth = model_depth.predict(X_test)
    acc_depth = accuracy_score(y_test, y_pred_depth)
    print(f"max_depth={depth}: Accuracy={acc_depth:.2f}")

print("\nscikit-learn 决策树 Demo完成！")
