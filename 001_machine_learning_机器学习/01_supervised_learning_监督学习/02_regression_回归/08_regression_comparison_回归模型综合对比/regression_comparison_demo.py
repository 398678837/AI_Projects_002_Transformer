import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost 未安装，将跳过 XGBoost 部分")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM 未安装，将跳过 LightGBM 部分")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost 未安装，将跳过 CatBoost 部分")

try:
    from sklearn.neural_network import MLPRegressor
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False
    print("警告: MLPRegressor 不可用")

def main():
    print("=" * 80)
    print("回归模型综合对比演示")
    print("=" * 80)
    
    # 加载数据
    print("\n1. 加载加州房价数据集")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    print(f"数据集形状: {X.shape}")
    print(f"特征数量: {X.shape[1]}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 定义模型
    models = {
        '线性回归': LinearRegression(),
        '岭回归': Ridge(alpha=1.0),
        'LASSO': Lasso(alpha=0.1),
        '弹性网络': ElasticNet(alpha=0.1, l1_ratio=0.5),
        '多项式回归(2次)': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ]),
        '决策树回归': DecisionTreeRegressor(max_depth=5, random_state=42),
        '随机森林回归': RandomForestRegressor(n_estimators=100, random_state=42),
        'GBDT回归': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR(RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ])
    }
    
    # 添加XGBoost、LightGBM、CatBoost
    if XGBOOST_AVAILABLE:
        models['XGBoost回归'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    if LIGHTGBM_AVAILABLE:
        models['LightGBM回归'] = LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
    if CATBOOST_AVAILABLE:
        models['CatBoost回归'] = CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
    
    # 添加神经网络
    if MLP_AVAILABLE:
        models['神经网络回归'] = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
        ])
    
    # 训练和评估所有模型
    print("\n" + "=" * 80)
    print("训练和评估模型")
    print("=" * 80)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练: {name}...")
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'r2_train': r2_train,
            'r2_test': r2_test
        }
        
        print(f"  训练集 - MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
        print(f"  测试集 - MSE: {mse_test:.4f}, R²: {r2_test:.4f}")
    
    # 可视化结果
    print("\n" + "=" * 80)
    print("可视化结果")
    print("=" * 80)
    
    model_names = list(results.keys())
    mse_train_values = [results[name]['mse_train'] for name in model_names]
    mse_test_values = [results[name]['mse_test'] for name in model_names]
    r2_train_values = [results[name]['r2_train'] for name in model_names]
    r2_test_values = [results[name]['r2_test'] for name in model_names]
    
    # 设置图形
    plt.figure(figsize=(18, 12))
    
    # MSE对比
    plt.subplot(2, 2, 1)
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width/2, mse_train_values, width, label='训练集', alpha=0.8)
    plt.bar(x + width/2, mse_test_values, width, label='测试集', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('MSE')
    plt.title('Model MSE Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # R²对比
    plt.subplot(2, 2, 2)
    plt.bar(x - width/2, r2_train_values, width, label='训练集', alpha=0.8)
    plt.bar(x + width/2, r2_test_values, width, label='测试集', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('Model R² Score Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 1])
    
    # 测试集R²排序
    plt.subplot(2, 2, 3)
    sorted_indices = np.argsort(r2_test_values)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_r2_test = [r2_test_values[i] for i in sorted_indices]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_names)))
    bars = plt.barh(sorted_names, sorted_r2_test, color=colors)
    plt.xlabel('R² Score')
    plt.title('Test Set R² Score Ranking')
    plt.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center')
    plt.xlim([0, max(sorted_r2_test) * 1.1])
    
    # 训练集vs测试集R²
    plt.subplot(2, 2, 4)
    plt.scatter(r2_train_values, r2_test_values, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        plt.annotate(name, (r2_train_values[i], r2_test_values[i]), 
                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    plt.xlabel('Training Set R²')
    plt.ylabel('Test Set R²')
    plt.title('Training Set vs Test Set R²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('images/regression_models_comparison.png', dpi=300, bbox_inches='tight')
    print("\n回归模型对比图已保存到 images/regression_models_comparison.png")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    print(f"\n最佳模型（测试集R²最高）: {sorted_names[0]}")
    print(f"R² Score: {sorted_r2_test[0]:.4f}")
    
    print("\n" + "=" * 80)
    print("所有模型测试集R²排序:")
    print("=" * 80)
    for i, name in enumerate(sorted_names):
        print(f"{i+1}. {name}: {sorted_r2_test[i]:.4f}")
    
    print("\n" + "=" * 80)
    print("回归模型综合对比演示完成！")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    main()
