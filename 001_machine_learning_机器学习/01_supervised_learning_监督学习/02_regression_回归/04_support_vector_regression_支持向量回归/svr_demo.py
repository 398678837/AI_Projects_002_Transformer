import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 60)
    print("支持向量回归（SVR）演示")
    print("=" * 60)
    
    # 第一步：生成模拟的非线性数据
    print("\n1. 生成模拟的非线性数据")
    np.random.seed(42)
    X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
    y_simple = np.sin(X_simple) + np.random.normal(0, 0.1, size=X_simple.shape)
    
    print(f"模拟数据形状: {X_simple.shape}")
    
    # 第二步：使用不同核函数的SVR
    print("\n" + "=" * 60)
    print("使用不同核函数的SVR")
    print("=" * 60)
    
    # 标准化数据（SVR对特征尺度敏感）
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_simple_scaled = scaler_X.fit_transform(X_simple)
    y_simple_scaled = scaler_y.fit_transform(y_simple).ravel()
    
    # 定义不同的SVR模型
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    models = []
    
    plt.figure(figsize=(16, 10))
    
    for i, kernel in enumerate(kernels, 1):
        # 创建SVR模型
        if kernel == 'poly':
            model = SVR(kernel=kernel, degree=3, C=1.0, epsilon=0.1)
        else:
            model = SVR(kernel=kernel, C=1.0, epsilon=0.1)
        
        # 训练
        model.fit(X_simple_scaled, y_simple_scaled)
        
        # 预测
        y_pred_scaled = model.predict(X_simple_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # 评估
        mse = mean_squared_error(y_simple, y_pred)
        r2 = r2_score(y_simple, y_pred)
        
        models.append(model)
        
        # 绘制结果
        plt.subplot(2, 2, i)
        plt.scatter(X_simple, y_simple, alpha=0.5, label='真实数据')
        plt.plot(X_simple, y_pred, 'r-', linewidth=2, label=f'SVR ({kernel})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Kernel: {kernel}\nMSE: {mse:.4f}, R²: {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/svr_kernels.png', dpi=300, bbox_inches='tight')
    print("\n不同核函数的SVR对比图已保存到 images/svr_kernels.png")
    
    # 第三步：使用真实数据（加州房价数据集）
    print("\n" + "=" * 60)
    print("使用加州房价数据集进行SVR")
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
    
    # 标准化数据
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    # 创建和训练SVR模型（使用RBF核）
    print(f"\n使用RBF核函数的SVR")
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train_scaled)
    
    # 预测
    y_pred_train_scaled = svr.predict(X_train_scaled)
    y_pred_test_scaled = svr.predict(X_test_scaled)
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
    
    # 评估
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n训练集 MSE: {mse_train:.4f}")
    print(f"训练集 R²: {r2_train:.4f}")
    print(f"测试集 MSE: {mse_test:.4f}")
    print(f"测试集 R²: {r2_test:.4f}")
    
    # 获取支持向量
    print(f"\n支持向量数量: {len(svr.support_)}")
    print(f"支持向量索引: {svr.support_[:10]}...")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 训练数据
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='训练数据')
    # 绘制支持向量
    plt.scatter(X_train[svr.support_], y_train[svr.support_], 
                s=100, linewidth=1, facecolors='none', edgecolors='k', label='支持向量')
    X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    X_plot_scaled = scaler_X.transform(X_plot)
    y_plot_scaled = svr.predict(X_plot_scaled)
    y_plot = scaler_y.inverse_transform(y_plot_scaled.reshape(-1, 1)).ravel()
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='SVR (RBF)')
    plt.xlabel('Median Income (MedInc)')
    plt.ylabel('Median House Price')
    plt.title(f'训练集 - SVR (RBF)\nR² = {r2_train:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 测试数据
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, alpha=0.5, label='测试数据')
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='SVR (RBF)')
    plt.xlabel('Median Income (MedInc)')
    plt.ylabel('Median House Price')
    plt.title(f'测试集 - SVR (RBF)\nR² = {r2_test:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/svr_california.png', dpi=300, bbox_inches='tight')
    print("\n加州房价数据集SVR图已保存到 images/svr_california.png")
    
    # 第四步：使用网格搜索调优SVR参数
    print("\n" + "=" * 60)
    print("使用网格搜索调优SVR参数")
    print("=" * 60)
    
    # 使用多个特征（前3个特征）
    X_multi = X[:, :3]
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler_X_multi = StandardScaler()
    scaler_y_multi = StandardScaler()
    X_train_multi_scaled = scaler_X_multi.fit_transform(X_train_multi)
    X_test_multi_scaled = scaler_X_multi.transform(X_test_multi)
    y_train_multi_scaled = scaler_y_multi.fit_transform(y_train_multi.reshape(-1, 1)).ravel()
    
    # 定义参数网格
    param_grid = {
        'kernel': ['rbf'],
        'C': [0.1, 1.0, 10.0],
        'epsilon': [0.01, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.1, 1.0]
    }
    
    print(f"\n参数网格: {param_grid}")
    
    # 创建网格搜索
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_multi_scaled, y_train_multi_scaled)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证R²: {grid_search.best_score_:.4f}")
    
    # 使用最佳模型
    best_svr = grid_search.best_estimator_
    
    # 预测
    y_pred_train_best_scaled = best_svr.predict(X_train_multi_scaled)
    y_pred_test_best_scaled = best_svr.predict(X_test_multi_scaled)
    y_pred_train_best = scaler_y_multi.inverse_transform(y_pred_train_best_scaled.reshape(-1, 1)).ravel()
    y_pred_test_best = scaler_y_multi.inverse_transform(y_pred_test_best_scaled.reshape(-1, 1)).ravel()
    
    # 评估
    mse_train_best = mean_squared_error(y_train_multi, y_pred_train_best)
    mse_test_best = mean_squared_error(y_test_multi, y_pred_test_best)
    r2_train_best = r2_score(y_train_multi, y_pred_train_best)
    r2_test_best = r2_score(y_test_multi, y_pred_test_best)
    
    print(f"\n最佳模型训练集 MSE: {mse_train_best:.4f}")
    print(f"最佳模型训练集 R²: {r2_train_best:.4f}")
    print(f"最佳模型测试集 MSE: {mse_test_best:.4f}")
    print(f"最佳模型测试集 R²: {r2_test_best:.4f}")
    
    # 可视化C和epsilon的影响
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    epsilon_values = [0.01, 0.1, 0.2, 0.5]
    
    train_r2_scores = np.zeros((len(C_values), len(epsilon_values)))
    test_r2_scores = np.zeros((len(C_values), len(epsilon_values)))
    
    for i, C in enumerate(C_values):
        for j, epsilon in enumerate(epsilon_values):
            svr_temp = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma='scale')
            svr_temp.fit(X_train_multi_scaled, y_train_multi_scaled)
            train_r2_scores[i, j] = r2_score(y_train_multi, scaler_y_multi.inverse_transform(svr_temp.predict(X_train_multi_scaled).reshape(-1, 1)).ravel())
            test_r2_scores[i, j] = r2_score(y_test_multi, scaler_y_multi.inverse_transform(svr_temp.predict(X_test_multi_scaled).reshape(-1, 1)).ravel())
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for j, epsilon in enumerate(epsilon_values):
        plt.semilogx(C_values, train_r2_scores[:, j], 'o-', label=f'epsilon={epsilon}')
    plt.xlabel('C')
    plt.ylabel('R² Score')
    plt.title('Training Set - C vs R² Score')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    plt.subplot(1, 2, 2)
    for j, epsilon in enumerate(epsilon_values):
        plt.semilogx(C_values, test_r2_scores[:, j], 's-', label=f'epsilon={epsilon}')
    plt.xlabel('C')
    plt.ylabel('R² Score')
    plt.title('Test Set - C vs R² Score')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('images/svr_parameters_effect.png', dpi=300, bbox_inches='tight')
    print("\nSVR参数影响图已保存到 images/svr_parameters_effect.png")
    
    print("\n" + "=" * 60)
    print("支持向量回归演示完成！")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()
