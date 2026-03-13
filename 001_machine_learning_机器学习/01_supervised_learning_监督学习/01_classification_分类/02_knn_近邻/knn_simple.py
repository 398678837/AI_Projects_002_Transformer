"""
KNN（K最近邻）算法的纯Python实现
不依赖任何外部库，适合初学者理解核心原理
"""

import numpy as np

class KNN:
    """K最近邻分类器"""
    
    def __init__(self, k=3):
        """初始化KNN分类器
        
        Args:
            k: 近邻数量
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1, x2):
        """计算欧几里得距离
        
        Args:
            x1: 第一个样本
            x2: 第二个样本
            
        Returns:
            两个样本之间的欧几里得距离
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X, y):
        """训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """预测新样本
        
        Args:
            X: 测试特征
            
        Returns:
            预测标签
        """
        predictions = []
        for x in X:
            # 计算与所有训练样本的距离
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # 获取K个最近邻的索引
            k_indices = np.argsort(distances)[:self.k]
            
            # 获取K个最近邻的标签
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # 多数投票
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)
        
        return predictions

def accuracy(y_true, y_pred):
    """计算准确率"""
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    """计算混淆矩阵"""
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    
    return matrix

# 1. 创建简单的分类数据集
print("创建数据集...")
# 特征：[x1, x2]
X = np.array([
    [1.0, 2.0],   # 类别0
    [1.5, 1.8],   # 类别0
    [5.0, 8.0],   # 类别1
    [8.0, 8.0],   # 类别1
    [1.0, 0.6],   # 类别0
    [9.0, 11.0],  # 类别1
    [8.0, 2.0],   # 类别2
    [10.0, 2.0],  # 类别2
    [9.0, 3.0],   # 类别2
])

# 标签
y = np.array([0, 0, 1, 1, 0, 1, 2, 2, 2])

print(f"数据集形状: X={X.shape}, y={y.shape}")
print("特征:")
print(X)
print("标签:")
print(y)

# 2. 划分训练集和测试集
print("\n划分训练集和测试集...")
X_train, X_test = X[:6], X[6:]
y_train, y_test = y[:6], y[6:]
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 3. 创建并训练模型
print("\n训练KNN模型...")
model = KNN(k=3)
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

# 6. 预测示例
print("\n预测示例:")
samples = np.array([[2.0, 3.0], [6.0, 7.0], [8.5, 2.5]])
predictions = model.predict(samples)

for i, sample in enumerate(samples):
    print(f"样本 {sample} → 预测类别: {predictions[i]}")

print("\n纯Python KNN Demo完成！")
