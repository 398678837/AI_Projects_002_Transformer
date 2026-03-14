import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline

def main():
    print("=" * 60)
    print("多项式回归演示")
    print("=" * 60)
    
    # 第一步：生成模拟的非线性数据
    print("\n1. 生成模拟的非线性数据")
    np.random.seed(42)
    X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
    y_simple = np.sin(X_simple) + np.random.normal(0, 0.1, size=X_simple.shape)
    
    print(f"模拟数据形状: {X_simple.shape}")
    
    # 第二步：使用不同次数的多项式进行拟合
    degrees = [1, 2, 3, 5, 10]
    
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees, 1):
        # 创建多项式特征
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_simple)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_poly, y_simple)
        
        # 预测
        y_pred = model.predict(X_poly)
        
        # 评估
        mse = mean_squared_error(y_simple, y_pred)
        r2 = r2_score(y_simple, y_pred)
        
        # 绘制结果
        plt.subplot(2, 3, i)
        plt.scatter(X_simple, y_simple, alpha=0.5, label='真实数据')
        plt.plot(X_simple, y_pred, 'r-', linewidth=2, label=f'多项式回归 (degree={degree})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Degree {degree}\nMSE: {mse:.4f}, R²: {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/polynomial_regression_degrees.png', dpi=300, bbox_inches='tight')
    print("\n不同次数的多项式回归对比图已保存到 images/polynomial_regression_degrees.png")
    
    # 第三步：使用真实数据（加州房价数据集）
    print("\n" + "=" * 60)
    print("使用加州房价数据集进行多项式回归")
    print("=" * 60)
    
    # 加载数据
    print("\n1. 加载加州房价数据集")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names
    
    print(f"数据集形状: {X.shape}")
    print(f"特征名称: {feature_names}")
    print(f"目标变量: 房价中位数 (单位: 10万美元)")
    
    # 为了演示，我们只使用一个特征（MedInc - 收入中位数）
    X_single = X[:, 0].reshape(-1, 1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 使用Pipeline进行多项式回归
    degree = 2
    print(f"\n使用 {degree} 次多项式回归")
    
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    # 训练
    pipeline.fit(X_train, y_train)
    
    # 预测
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # 评估
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n训练集 MSE: {mse_train:.4f}")
    print(f"训练集 R²: {r2_train:.4f}")
    print(f"测试集 MSE: {mse_test:.4f}")
    print(f"测试集 R²: {r2_test:.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 训练数据
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='训练数据')
    X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_plot = pipeline.predict(X_plot)
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'{degree}次多项式回归')
    plt.xlabel('Median Income (MedInc)')
    plt.ylabel('Median House Price')
    plt.title(f'训练集 - {degree}次多项式回归\nR² = {r2_train:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 测试数据
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, alpha=0.5, label='测试数据')
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'{degree}次多项式回归')
    plt.xlabel('Median Income (MedInc)')
    plt.ylabel('Median House Price')
    plt.title(f'测试集 - {degree}次多项式回归\nR² = {r2_test:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/polynomial_regression_california.png', dpi=300, bbox_inches='tight')
    print("\n加州房价数据集多项式回归图已保存到 images/polynomial_regression_california.png")
    
    # 第四步：使用多个特征进行多项式回归
    print("\n" + "=" * 60)
    print("使用多个特征进行多项式回归")
    print("=" * 60)
    
    # 使用前3个特征
    X_multi = X[:, :3]
    
    # 划分训练集和测试集
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y, test_size=0.2, random_state=42
    )
    
    # 使用Pipeline
    degree = 2
    print(f"\n使用 {degree} 次多项式回归（3个特征）")
    
    pipeline_multi = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    # 训练
    pipeline_multi.fit(X_train_multi, y_train_multi)
    
    # 预测
    y_pred_train_multi = pipeline_multi.predict(X_train_multi)
    y_pred_test_multi = pipeline_multi.predict(X_test_multi)
    
    # 评估
    mse_train_multi = mean_squared_error(y_train_multi, y_pred_train_multi)
    mse_test_multi = mean_squared_error(y_test_multi, y_pred_test_multi)
    r2_train_multi = r2_score(y_train_multi, y_pred_train_multi)
    r2_test_multi = r2_score(y_test_multi, y_pred_test_multi)
    
    print(f"\n训练集 MSE: {mse_train_multi:.4f}")
    print(f"训练集 R²: {r2_train_multi:.4f}")
    print(f"测试集 MSE: {mse_test_multi:.4f}")
    print(f"测试集 R²: {r2_test_multi:.4f}")
    
    # 第五步：正则化多项式回归
    print("\n" + "=" * 60)
    print("正则化多项式回归（Ridge）")
    print("=" * 60)
    
    from sklearn.linear_model import Ridge
    
    # 创建Pipeline
    pipeline_ridge = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    # 训练
    pipeline_ridge.fit(X_train_multi, y_train_multi)
    
    # 预测
    y_pred_train_ridge = pipeline_ridge.predict(X_train_multi)
    y_pred_test_ridge = pipeline_ridge.predict(X_test_multi)
    
    # 评估
    mse_train_ridge = mean_squared_error(y_train_multi, y_pred_train_ridge)
    mse_test_ridge = mean_squared_error(y_test_multi, y_pred_test_ridge)
    r2_train_ridge = r2_score(y_train_multi, y_pred_train_ridge)
    r2_test_ridge = r2_score(y_test_multi, y_pred_test_ridge)
    
    print(f"\nRidge回归训练集 MSE: {mse_train_ridge:.4f}")
    print(f"Ridge回归训练集 R²: {r2_train_ridge:.4f}")
    print(f"Ridge回归测试集 MSE: {mse_test_ridge:.4f}")
    print(f"Ridge回归测试集 R²: {r2_test_ridge:.4f}")
    
    # 可视化不同alpha值的效果
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_r2_scores = []
    test_r2_scores = []
    
    for alpha in alphas:
        pipeline_ridge_alpha = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=alpha))
        ])
        pipeline_ridge_alpha.fit(X_train_multi, y_train_multi)
        train_r2_scores.append(r2_score(y_train_multi, pipeline_ridge_alpha.predict(X_train_multi)))
        test_r2_scores.append(r2_score(y_test_multi, pipeline_ridge_alpha.predict(X_test_multi)))
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, train_r2_scores, 'o-', label='训练集')
    plt.semilogx(alphas, test_r2_scores, 's-', label='测试集')
    plt.xlabel('alpha')
    plt.ylabel('R² Score')
    plt.title('Ridge Regression - alpha vs R² Score')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig('images/ridge_alpha_effect.png', dpi=300, bbox_inches='tight')
    print("\nRidge回归alpha值影响图已保存到 images/ridge_alpha_effect.png")
    
    print("\n" + "=" * 60)
    print("多项式回归演示完成！")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()
