import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 80)
    print("神经网络回归（DNN）演示")
    print("=" * 80)
    
    # 第一步：生成模拟的非线性数据
    print("\n1. 生成模拟的非线性数据")
    np.random.seed(42)
    X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
    y_simple = np.sin(X_simple) + np.random.normal(0, 0.1, size=X_simple.shape)
    
    print(f"模拟数据形状: {X_simple.shape}")
    
    # 第二步：使用不同结构的神经网络
    print("\n" + "=" * 80)
    print("使用不同结构的神经网络")
    print("=" * 80)
    
    # 标准化数据
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_simple_scaled = scaler_X.fit_transform(X_simple)
    y_simple_scaled = scaler_y.fit_transform(y_simple).ravel()
    
    # 定义不同的神经网络结构
    architectures = [
        (10,),
        (50,),
        (100,),
        (100, 50),
        (100, 50, 25)
    ]
    architecture_names = [
        '1层(10)',
        '1层(50)',
        '1层(100)',
        '2层(100,50)',
        '3层(100,50,25)'
    ]
    
    models = []
    
    plt.figure(figsize=(18, 12))
    
    for i, (hidden_layers, name) in enumerate(zip(architectures, architecture_names), 1):
        # 创建神经网络模型
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        
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
        plt.subplot(2, 3, i)
        plt.scatter(X_simple, y_simple, alpha=0.5, label='真实数据')
        plt.plot(X_simple, y_pred, 'r-', linewidth=2, label=f'NN {name}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'结构: {name}\nMSE: {mse:.4f}, R²: {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/nn_architectures.png', dpi=300, bbox_inches='tight')
    print("\n不同结构的神经网络对比图已保存到 images/nn_architectures.png")
    
    # 第三步：使用真实数据（加州房价数据集）
    print("\n" + "=" * 80)
    print("使用加州房价数据集进行神经网络回归")
    print("=" * 80)
    
    # 加载数据
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
    
    # 标准化数据
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    # 创建和训练神经网络
    print(f"\n使用两层神经网络（100, 50）")
    nn = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
    nn.fit(X_train_scaled, y_train_scaled)
    
    # 预测
    y_pred_train_scaled = nn.predict(X_train_scaled)
    y_pred_test_scaled = nn.predict(X_test_scaled)
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
    
    # 可视化训练曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(nn.loss_curve_, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 预测vs真实值
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'预测vs真实值\nR² = {r2_test:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/nn_training_california.png', dpi=300, bbox_inches='tight')
    print("\n加州房价数据集神经网络训练图已保存到 images/nn_training_california.png")
    
    # 第四步：不同激活函数的影响
    print("\n" + "=" * 80)
    print("不同激活函数的影响")
    print("=" * 80)
    
    activations = ['identity', 'logistic', 'tanh', 'relu']
    activation_names = ['恒等', 'Logistic', 'Tanh', 'ReLU']
    
    train_r2_scores = []
    test_r2_scores = []
    
    for activation in activations:
        nn_act = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation=activation,
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        nn_act.fit(X_train_scaled, y_train_scaled)
        train_r2_scores.append(r2_score(y_train, scaler_y.inverse_transform(nn_act.predict(X_train_scaled).reshape(-1, 1)).ravel()))
        test_r2_scores.append(r2_score(y_test, scaler_y.inverse_transform(nn_act.predict(X_test_scaled).reshape(-1, 1)).ravel()))
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(activation_names))
    width = 0.35
    plt.bar(x - width/2, train_r2_scores, width, label='训练集', alpha=0.8)
    plt.bar(x + width/2, test_r2_scores, width, label='测试集', alpha=0.8)
    plt.xlabel('Activation Function')
    plt.ylabel('R² Score')
    plt.title('R² Score Comparison for Different Activation Functions')
    plt.xticks(x, activation_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 1])
    plt.savefig('images/nn_activations.png', dpi=300, bbox_inches='tight')
    print("\n不同激活函数对比图已保存到 images/nn_activations.png")
    
    print("\n" + "=" * 80)
    print("神经网络回归演示完成！")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    main()
