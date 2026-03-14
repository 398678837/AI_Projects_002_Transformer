"""
线性回归（Linear Regression）算法的scikit-learn实现
使用波士顿房价数据集进行回归预测
"""

import numpy as np
from sklearn.linear_model import LinearRegression
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

# 4. 创建并训练模型
print("\n训练线性回归模型...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 模型预测
print("\n预测测试集...")
y_pred = model.predict(X_test)

# 6. 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"R²评分: {r2:.4f}")

# 7. 查看模型参数
print("\n模型参数:")
print(f" 截距 (intercept): {model.intercept_:.4f}")
print(" 系数 (coefficients):")
for i, (feature, coef) in enumerate(zip(boston.feature_names, model.coef_)):
    print(f"    {feature}: {coef:.4f}")

# 8. 特征重要性分析（基于系数绝对值）
print("\n特征重要性（基于系数绝对值）:")
feature_importance = np.abs(model.coef_)
for i, (feature, importance) in enumerate(zip(boston.feature_names, feature_importance)):
    print(f"{feature}: {importance:.4f}")

# 9. 可视化预测结果
print("\n可视化预测结果...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('实际房价')
plt.ylabel('预测房价')
plt.title('线性回归：实际房价 vs 预测房价')
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('linear_regression_predictions.png')
print("预测结果可视化已保存为 'linear_regression_predictions.png'")

# 10. 简单的预测示例
print("\n预测示例:")
# 随机选择一个测试样本
sample_idx = 0
sample = X_test[sample_idx]
actual_price = y_test[sample_idx]
predicted_price = y_pred[sample_idx]

print(f"样本特征: {sample}")
print(f"实际房价: {actual_price:.2f}")
print(f"预测房价: {predicted_price:.2f}")
print(f"预测误差: {abs(predicted_price - actual_price):.2f}")

# 11. 残差分析
print("\n残差分析...")
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测房价')
plt.ylabel('残差')
plt.title('线性回归：残差分析')
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('linear_regression_residuals.png')
print("残差分析可视化已保存为 'linear_regression_residuals.png'")

print("\nscikit-learn 线性回归 Demo完成！")
