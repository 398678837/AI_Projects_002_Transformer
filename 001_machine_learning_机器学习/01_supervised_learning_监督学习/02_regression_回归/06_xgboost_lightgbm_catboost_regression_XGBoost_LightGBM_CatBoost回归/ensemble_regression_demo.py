import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost 未安装")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM 未安装")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost 未安装")

def main():
    print("=" * 80)
    print("XGBoost/LightGBM/CatBoost回归演示")
    print("=" * 80)
    
    # 第一步：加载数据
    print("\n1. 加载加州房价数据集")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names
    
    print(f"数据集形状: {X.shape}")
    print(f"特征名称: {feature_names}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 第二步：定义和训练模型
    models = {}
    results = {}
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("\n训练 XGBoost...")
        xgb = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb
        
        y_pred_train = xgb.predict(X_train)
        y_pred_test = xgb.predict(X_test)
        
        results['XGBoost'] = {
            'mse_train': mean_squared_error(y_train, y_pred_train),
            'mse_test': mean_squared_error(y_test, y_pred_test),
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'feature_importances': xgb.feature_importances_
        }
        
        print(f"  训练集 - MSE: {results['XGBoost']['mse_train']:.4f}, R²: {results['XGBoost']['r2_train']:.4f}")
        print(f"  测试集 - MSE: {results['XGBoost']['mse_test']:.4f}, R²: {results['XGBoost']['r2_test']:.4f}")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n训练 LightGBM...")
        lgb = LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        lgb.fit(X_train, y_train)
        models['LightGBM'] = lgb
        
        y_pred_train = lgb.predict(X_train)
        y_pred_test = lgb.predict(X_test)
        
        results['LightGBM'] = {
            'mse_train': mean_squared_error(y_train, y_pred_train),
            'mse_test': mean_squared_error(y_test, y_pred_test),
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'feature_importances': lgb.feature_importances_
        }
        
        print(f"  训练集 - MSE: {results['LightGBM']['mse_train']:.4f}, R²: {results['LightGBM']['r2_train']:.4f}")
        print(f"  测试集 - MSE: {results['LightGBM']['mse_test']:.4f}, R²: {results['LightGBM']['r2_test']:.4f}")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        print("\n训练 CatBoost...")
        cat = CatBoostRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        cat.fit(X_train, y_train)
        models['CatBoost'] = cat
        
        y_pred_train = cat.predict(X_train)
        y_pred_test = cat.predict(X_test)
        
        results['CatBoost'] = {
            'mse_train': mean_squared_error(y_train, y_pred_train),
            'mse_test': mean_squared_error(y_test, y_pred_test),
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'feature_importances': cat.feature_importances_
        }
        
        print(f"  训练集 - MSE: {results['CatBoost']['mse_train']:.4f}, R²: {results['CatBoost']['r2_train']:.4f}")
        print(f"  测试集 - MSE: {results['CatBoost']['mse_test']:.4f}, R²: {results['CatBoost']['r2_test']:.4f}")
    
    if not results:
        print("\n错误: 没有可用的模型，请安装 XGBoost, LightGBM 或 CatBoost")
        return
    
    # 第三步：可视化结果
    print("\n" + "=" * 80)
    print("可视化结果")
    print("=" * 80)
    
    model_names = list(results.keys())
    n_models = len(model_names)
    
    plt.figure(figsize=(18, 12))
    
    # R²对比
    plt.subplot(2, 3, 1)
    x = np.arange(n_models)
    width = 0.35
    r2_train_values = [results[name]['r2_train'] for name in model_names]
    r2_test_values = [results[name]['r2_test'] for name in model_names]
    plt.bar(x - width/2, r2_train_values, width, label='训练集', alpha=0.8)
    plt.bar(x + width/2, r2_test_values, width, label='测试集', alpha=0.8)
    plt.xlabel('模型')
    plt.ylabel('R² Score')
    plt.title('模型R² Score对比')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 1])
    
    # MSE对比
    plt.subplot(2, 3, 2)
    mse_train_values = [results[name]['mse_train'] for name in model_names]
    mse_test_values = [results[name]['mse_test'] for name in model_names]
    plt.bar(x - width/2, mse_train_values, width, label='训练集', alpha=0.8)
    plt.bar(x + width/2, mse_test_values, width, label='测试集', alpha=0.8)
    plt.xlabel('模型')
    plt.ylabel('MSE')
    plt.title('模型MSE对比')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 特征重要性对比
    for i, name in enumerate(model_names):
        plt.subplot(2, 3, i + 3)
        importances = results[name]['feature_importances']
        indices = np.argsort(importances)[::-1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        plt.barh(range(len(indices)), importances[indices], color=colors, alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('特征重要性')
        plt.title(f'{name} 特征重要性')
        plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('001_machine_learning_机器学习/01_supervised_learning_监督学习/02_regression_回归/06_xgboost_lightgbm_catboost_regression_XGBoost_LightGBM_CatBoost回归/images/ensemble_regression_comparison.png', dpi=300, bbox_inches='tight')
    print("\n集成回归模型对比图已保存到 images/ensemble_regression_comparison.png")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    sorted_by_r2 = sorted(results.items(), key=lambda x: x[1]['r2_test'], reverse=True)
    print(f"\n最佳模型（测试集R²最高）: {sorted_by_r2[0][0]}")
    print(f"R² Score: {sorted_by_r2[0][1]['r2_test']:.4f}")
    
    print("\n" + "=" * 80)
    print("所有模型测试集R²排序:")
    print("=" * 80)
    for i, (name, result) in enumerate(sorted_by_r2):
        print(f"{i+1}. {name}: R² = {result['r2_test']:.4f}, MSE = {result['mse_test']:.4f}")
    
    print("\n" + "=" * 80)
    print("XGBoost/LightGBM/CatBoost回归演示完成！")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    main()
