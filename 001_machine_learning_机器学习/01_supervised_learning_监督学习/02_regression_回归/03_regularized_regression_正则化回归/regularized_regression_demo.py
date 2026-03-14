"""
正则化回归（岭回归、LASSO回归、弹性网络）的scikit-learn实现
使用波士顿房价数据集进行回归预测
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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
    
    # 查看模型参数
    print(f"  截距 (intercept): {model.intercept_:.4f}")
    print("  系数 (coefficients):")
    for i, (feature, coef) in enumerate(zip(boston.feature_names, model.coef_)):
        print(f"    {feature}: {coef:.4f}")
    
    return model, y_pred, r2

# 5. 训练和评估不同的正则化回归模型
models = {}
y_preds = {}
r2_scores = {}

# 岭回归
ridge_model = Ridge(alpha=1.0, random_state=42)
models['岭回归'], y_preds['岭回归'], r2_scores['岭回归'] = train_and_evaluate(
    '岭回归', ridge_model, X_train, y_train, X_test, y_test
)

# LASSO回归
lasso_model = Lasso(alpha=0.1, random_state=42)
models['LASSO回归'], y_preds['LASSO回归'], r2_scores['LASSO回归'] = train_and_evaluate(
    'LASSO回归', lasso_model, X_train, y_train, X_test, y_test
)

# 弹性网络
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
models['弹性网络'], y_preds['弹性网络'], r2_scores['弹性网络'] = train_and_evaluate(
    '弹性网络', elastic_net_model, X_train, y_train, X_test, y_test
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
plt.savefig('regularized_regression_predictions.png')
print("预测结果可视化已保存为 'regularized_regression_predictions.png'")

# 8. 特征重要性分析
print("\n特征重要性分析...")
plt.figure(figsize=(15, 10))

for i, (model_name, model) in enumerate(models.items()):
    feature_importance = np.abs(model.coef_)
    plt.subplot(2, 2, i+1)
    plt.barh(boston.feature_names, feature_importance, color='skyblue')
    plt.xlabel('重要性（系数绝对值）')
    plt.ylabel('特征')
    plt.title(f'{model_name}特征重要性')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('regularized_regression_feature_importance.png')
print("特征重要性可视化已保存为 'regularized_regression_feature_importance.png'")

# 9. 正则化参数调优
print("\n正则化参数调优...")

# 测试不同的alpha值
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_r2 = []
lasso_r2 = []
elastic_net_r2 = []

for alpha in alphas:
    # 岭回归
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train, y_train)
    ridge_r2.append(r2_score(y_test, ridge.predict(X_test)))
    
    # LASSO回归
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_train, y_train)
    lasso_r2.append(r2_score(y_test, lasso.predict(X_test)))
    
    # 弹性网络
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X_train, y_train)
    elastic_net_r2.append(r2_score(y_test, elastic_net.predict(X_test)))

# 可视化不同alpha值的效果
plt.figure(figsize=(12, 6))
plt.plot(alphas, ridge_r2, 'o-', label='岭回归')
plt.plot(alphas, lasso_r2, 's-', label='LASSO回归')
plt.plot(alphas, elastic_net_r2, '^-', label='弹性网络')
plt.xscale('log')
plt.xlabel('alpha（正则化参数）')
plt.ylabel('R²评分')
plt.title('不同alpha值对模型性能的影响')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('regularized_regression_alpha_effect.png')
print("正则化参数调优可视化已保存为 'regularized_regression_alpha_effect.png'")

print("\nscikit-learn 正则化回归 Demo完成！")
