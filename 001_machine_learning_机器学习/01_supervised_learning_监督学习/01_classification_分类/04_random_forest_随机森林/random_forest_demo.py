"""
随机森林（Random Forest）算法的scikit-learn实现
使用鸢尾花数据集进行多分类
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
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
print(f"类别名称: {iris.target_names}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 3. 创建并训练模型
print("\n训练随机森林模型...")
# n_estimators: 树的数量
# max_depth: 每棵树的最大深度
# random_state: 随机种子
model = RandomForestClassifier(
    n_estimators=100,  # 100棵决策树
    max_depth=3,        # 每棵树的最大深度
    random_state=42     # 随机种子
)
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
print(f" 树的数量: {model.n_estimators}")
print(f"每棵树的最大深度: {model.max_depth}")

# 7. 特征重要性分析
print("\n特征重要性:")
feature_importance = model.feature_importances_
for i, (feature, importance) in enumerate(zip(iris.feature_names, feature_importance)):
    print(f"{feature}: {importance:.4f}")

# 8. 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(iris.feature_names, feature_importance, color='skyblue')
plt.title('随机森林特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('random_forest_feature_importance.png')
print("特征重要性可视化已保存为 'random_forest_feature_importance.png'")

# 9. 简单的预测示例
print("\n预测示例:")
# 随机选择一个测试样本
sample_idx = 0
sample = X_test[sample_idx]
actual_label = y_test[sample_idx]
predicted_label = y_pred[sample_idx]

print(f"样本特征: {sample}")
print(f"实际类别: {iris.target_names[actual_label]}")
print(f"预测类别: {iris.target_names[predicted_label]}")

# 10. 测试不同n_estimators的效果
print("\n测试不同n_estimators的效果:")
for n_trees in [10, 50, 100, 200, 300]:
    model_trees = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=3,
        random_state=42
    )
    model_trees.fit(X_train, y_train)
    y_pred_trees = model_trees.predict(X_test)
    acc_trees = accuracy_score(y_test, y_pred_trees)
    print(f"n_estimators={n_trees}: 准确率={acc_trees:.2f}")

print("\nscikit-learn 随机森林 Demo完成！")
