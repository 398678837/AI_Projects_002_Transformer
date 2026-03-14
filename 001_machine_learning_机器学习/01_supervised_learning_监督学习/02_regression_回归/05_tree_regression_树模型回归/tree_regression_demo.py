"""
树回归模型（决策树回归、随机森林回归、梯度提升树回归）的scikit-learn实现
使用波士顿房价数据集进行回归预测
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 加载数据集
print("加载加州房价数据集...")
from sklearn.datasets import fetch_california_housing
boston = fetch_california_housing()

X = boston.data  # 所有特征
y = boston.target  # 目标值（房价）

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {boston.feature_names}")

# 2. 数据预处理 - 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 4. 训练和评估函数
def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    """训练模型并评估性能"""
    print(f"\n训练{model_name}模型...")
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name}性能:")
    print(f"  均方误差 (MSE): {mse:.2f}")
    print(f"  均方根误差 (RMSE): {rmse:.2f}")
    print(f"  R²评分: {r2:.4f}")
    
    return model, y_pred, r2

# 5. 训练和评估不同的树回归模型
models = {}
y_preds = {}
r2_scores = {}

# 决策树回归
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
models['决策树回归'], y_preds['决策树回归'], r2_scores['决策树回归'] = train_and_evaluate(
    '决策树回归', dt_model, X_train, y_train, X_test, y_test
)

# 随机森林回归
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
models['随机森林回归'], y_preds['随机森林回归'], r2_scores['随机森林回归'] = train_and_evaluate(
    '随机森林回归', rf_model, X_train, y_train, X_test, y_test
)

# 梯度提升树回归
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
models['梯度提升树回归'], y_preds['梯度提升树回归'], r2_scores['梯度提升树回归'] = train_and_evaluate(
    '梯度提升树回归', gb_model, X_train, y_train, X_test, y_test
)

# 6. 模型性能对比
print("\n模型性能对比:")
for model_name, r2 in r2_scores.items():
    print(f"{model_name}: {r2:.4f}")

# 7. 可视化预测结果
print("\n可视化预测结果...")
plt.figure(figsize=(15, 10))

for i, (model_name, y_pred) in enumerate(y_preds.items()):
    plt.subplot(2, 2, i+1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('实际房价')
    plt.ylabel('预测房价')
    plt.title(f'{model_name}：实际房价 vs 预测房价')
    plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('tree_regression_predictions.png')
print("预测结果可视化已保存为 'tree_regression_predictions.png'")

# 8. 特征重要性分析
print("\n特征重要性分析...")
plt.figure(figsize=(15, 10))

for i, (model_name, model) in enumerate(models.items()):
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        plt.subplot(2, 2, i+1)
        plt.barh(boston.feature_names, feature_importance, color='skyblue')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.title(f'{model_name}特征重要性')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('tree_regression_feature_importance.png')
print("特征重要性可视化已保存为 'tree_regression_feature_importance.png'")

# 9. 决策树深度调优
print("\n决策树深度调优...")

depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
dt_r2 = []
rf_r2 = []

for depth in depths:
    # 决策树回归
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    dt_r2.append(r2_score(y_test, dt.predict(X_test)))
    
    # 随机森林回归
    rf = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42)
    rf.fit(X_train, y_train)
    rf_r2.append(r2_score(y_test, rf.predict(X_test)))

# 可视化不同深度的效果
plt.figure(figsize=(12, 6))
plt.plot(depths, dt_r2, 'o-', label='决策树回归')
plt.plot(depths, rf_r2, 's-', label='随机森林回归')
plt.xlabel('树的最大深度')
plt.ylabel('R²评分')
plt.title('不同树深度对模型性能的影响')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('tree_regression_depth_effect.png')
print("决策树深度调优可视化已保存为 'tree_regression_depth_effect.png'")

# 10. 随机森林树的数量调优
print("\n随机森林树的数量调优...")

n_estimators = [10, 50, 100, 200, 300, 500]
rf_r2 = []

for n in n_estimators:
    rf = RandomForestRegressor(n_estimators=n, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_r2.append(r2_score(y_test, rf.predict(X_test)))

# 可视化不同树数量的效果
plt.figure(figsize=(12, 6))
plt.plot(n_estimators, rf_r2, 'o-', label='随机森林回归')
plt.xlabel('树的数量')
plt.ylabel('R²评分')
plt.title('不同树数量对随机森林性能的影响')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('random_forest_n_estimators_effect.png')
print("随机森林树的数量调优可视化已保存为 'random_forest_n_estimators_effect.png'")

print("\nscikit-learn 树回归模型 Demo完成！")
