"""
纯Python实现的逻辑回归Demo
不依赖任何外部库，适合初学者理解核心原理
"""

import numpy as np

class LogisticRegression:
    """逻辑回归类"""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """训练模型"""
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for _ in range(self.epochs):
            # 线性组合
            linear_model = np.dot(X, self.weights) + self.bias
            # Sigmoid激活
            y_predicted = self.sigmoid(linear_model)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """预测"""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        # 二分类决策
        return [1 if i > 0.5 else 0 for i in y_predicted]

def accuracy(y_true, y_pred):
    """计算准确率"""
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    """计算混淆矩阵"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

# 1. 创建简单的二分类数据集
print("创建数据集...")
# 特征：[x1, x2]
X = np.array([
    [0.2, 0.3],  # 类别0
    [0.4, 0.5],  # 类别0
    [0.6, 0.7],  # 类别0
    [0.8, 0.9],  # 类别1
    [0.9, 0.8],  # 类别1
    [0.7, 0.6],  # 类别1
    [0.5, 0.4],  # 类别0
    [0.3, 0.2],  # 类别0
    [0.1, 0.1],  # 类别0
    [0.9, 0.9],  # 类别1
    [0.8, 0.7],  # 类别1
    [0.6, 0.8],  # 类别1
])

# 标签
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])

print(f"数据集形状: X={X.shape}, y={y.shape}")
print("特征:")
print(X)
print("标签:")
print(y)

# 2. 划分训练集和测试集
print("\n划分训练集和测试集...")
X_train, X_test = X[:8], X[8:]
y_train, y_test = y[:8], y[8:]
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 3. 创建并训练模型
print("\n训练逻辑回归模型...")
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# 4. 模型预测
print("\n预测测试集...")
y_pred = model.predict(X_test)

# 5. 模型评估
acc = accuracy(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n模型准确率: {acc:.2f}")
print("混淆矩阵:")
print(conf_matrix)

# 6. 查看模型参数
print("\n模型参数:")
print(f"系数: {model.weights}")
print(f"截距: {model.bias}")

# 7. 预测示例
print("\n预测示例:")
samples = np.array([[0.1, 0.2], [0.8, 0.8], [0.5, 0.5]])
predictions = model.predict(samples)

for i, sample in enumerate(samples):
    print(f"样本 {sample} → 预测类别: {predictions[i]}")

print("\n纯Python逻辑回归Demo完成！")
