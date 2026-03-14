# t-SNE降维教材

## 第一章：为什么需要t-SNE

### 1.1 高维数据的问题

高维数据很难直接可视化和理解。例如：
- 图像数据：28x28=784维
- 基因表达数据：可能有数千维
- 词向量：可能有数百维

### 1.2 t-SNE的优势

- **保持局部结构**：相近的点在低维空间也相近
- **可视化效果好**：能清晰展示数据中的聚类
- **非线性降维**：能处理非线性数据

## 第二章：t-SNE原理

### 2.1 基本思想

t-SNE通过概率分布来描述点之间的相似性：

1. **高维空间**：用高斯分布描述相似性
2. **低维空间**：用t-分布描述相似性
3. **优化目标**：最小化两个分布的差异

### 2.2 困惑度（perplexity）

困惑度可以理解为"每个点的有效邻居数"：
- **小值**（如5）：关注局部细节
- **中值**（如30）：平衡局部和全局
- **大值**（如50）：关注全局结构

**推荐范围**：5-50

### 2.3 学习率

学习率控制优化的步长：
- **太大**：结果像一个球
- **太小**：结果太分散
- **推荐值**：200左右

## 第三章：t-SNE实践

### 3.1 使用scikit-learn

```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建t-SNE模型
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

# 降维
X_tsne = tsne.fit_transform(X_scaled)
```

### 3.2 可视化

```python
import matplotlib.pyplot as plt

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.title('t-SNE降维结果')
plt.show()
```

## 第四章：t-SNE优缺点

### 4.1 优点

- 保持局部结构
- 可视化效果好
- 发现聚类
- 处理非线性数据

### 4.2 缺点

- 计算慢（O(n²)）
- 没有显式映射
- 参数敏感
- 随机结果
- 不保留全局结构

## 第五章：实用技巧

### 5.1 数据预处理

- **标准化**：必须的！
- **PCA预处理**：先降到50维再用t-SNE
- **样本量**：建议<10,000

### 5.2 参数调优

1. 尝试不同的perplexity
2. 调整learning_rate
3. 增加n_iter
4. 多次运行取最好

### 5.3 结果解读

- 观察聚类
- 检查分离度
- 寻找异常值
- 注意随机性

## 第六章：总结

### 6.1 核心要点

- t-SNE用于数据可视化
- 保持局部结构
- 需要调参
- 不适合大规模数据

### 6.2 学习路径

1. 理解基本原理
2. 实践t-SNE
3. 学习调参技巧
4. 在实际项目中应用

---

**练习题目**：

1. 用t-SNE可视化MNIST数据集的子集。
2. 尝试不同的perplexity，观察结果变化。
3. 先用PCA降维，再用t-SNE，比较结果。
4. 尝试不同的学习率，观察结果变化。
5. 用t-SNE可视化你自己的数据集。

**参考资料**：
- 《Visualizing Data using t-SNE》van der Maaten & Hinton
- scikit-learn官方文档
- t-SNE官方网站
