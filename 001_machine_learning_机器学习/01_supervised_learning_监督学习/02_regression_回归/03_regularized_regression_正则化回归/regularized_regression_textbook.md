# 正则化回归教材

## 第一章：正则化基础

### 1.1 为什么需要正则化

#### 1.1.1 过拟合问题
在机器学习中，当模型过于复杂时，容易发生过拟合：
- **训练集表现**：非常好，误差很小
- **测试集表现**：很差，泛化能力不足
- **原因**：模型学习了训练数据中的噪声，而不是真正的规律

#### 1.1.2 多重共线性
当自变量之间存在高度相关性时：
- **回归系数不稳定**：小的数据变化导致系数大幅变化
- **方差增大**：估计量的方差很大
- **解释困难**：难以解释各个变量的独立贡献

#### 1.1.3 高维数据
当特征数量远大于样本数量时：
- **解不唯一**：正规方程无解或无穷多解
- **计算困难**：矩阵求逆计算量大
- **过拟合风险**：模型复杂度过高

### 1.2 正则化的思想

正则化通过在目标函数中添加惩罚项来限制模型复杂度：

$$ \min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \cdot \text{Penalty}(\boldsymbol{\beta}) \right] $$

其中：
- $\lambda$ 是正则化强度参数
- $\text{Penalty}(\boldsymbol{\beta})$ 是惩罚函数

## 第二章：岭回归（Ridge Regression）

### 2.1 岭回归的原理

#### 2.1.1 L2正则化
岭回归使用L2范数作为惩罚项：

$$ \min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right] $$

#### 2.1.2 矩阵形式
$$ \min_{\boldsymbol{\beta}} \left[ (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \lambda \boldsymbol{\beta}^T\boldsymbol{\beta} \right] $$

### 2.2 岭回归的求解

#### 2.2.1 正规方程
岭回归的正规方程为：

$$ \boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y} $$

其中 $\mathbf{I}$ 是单位矩阵。

#### 2.2.2 几何解释
L2正则化相当于在参数空间中添加了一个球形约束：

$$ \sum_{j=1}^{p} \beta_j^2 \leq t $$

最优解位于残差平方和等高线与约束球的切点处。

### 2.3 岭回归的性质

#### 2.3.1 有偏估计
岭回归估计量是有偏的：

$$ E[\hat{\boldsymbol{\beta}}^{ridge}] \neq \boldsymbol{\beta} $$

#### 2.3.2 方差减小
虽然是有偏估计，但方差显著减小：

$$ \text{Var}(\hat{\boldsymbol{\beta}}^{ridge}) < \text{Var}(\hat{\boldsymbol{\beta}}^{OLS}) $$

#### 2.3.3 均方误差
在某些条件下，岭回归的均方误差小于OLS：

$$ \text{MSE}(\hat{\boldsymbol{\beta}}^{ridge}) < \text{MSE}(\hat{\boldsymbol{\beta}}^{OLS}) $$

### 2.4 岭参数的选择

#### 2.4.1 交叉验证
使用K折交叉验证选择最优的$\lambda$：

```python
from sklearn.linear_model import RidgeCV

# 创建模型
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)

# 训练模型
ridge_cv.fit(X_train, y_train)

# 最优alpha
best_alpha = ridge_cv.alpha_
```

#### 2.4.2 岭迹图
绘制不同$\lambda$值下回归系数的变化：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

alphas = np.logspace(-4, 4, 100)
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge Trace')
plt.show()
```

## 第三章：LASSO回归

### 3.1 LASSO的原理

#### 3.1.1 L1正则化
LASSO（Least Absolute Shrinkage and Selection Operator）使用L1范数作为惩罚项：

$$ \min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right] $$

#### 3.1.2 稀疏性
L1正则化会产生稀疏解，即某些系数精确为零：
- **特征选择**：自动进行特征选择
- **模型简化**：简化模型，提高可解释性
- **计算效率**：减少存储和计算成本

### 3.2 LASSO的求解

#### 3.2.1 坐标下降法
LASSO没有闭式解，需要使用迭代算法：

```python
from sklearn.linear_model import Lasso

# 创建模型
lasso = Lasso(alpha=0.1, max_iter=10000)

# 训练模型
lasso.fit(X_train, y_train)

# 预测
y_pred = lasso.predict(X_test)
```

#### 3.2.2 最小角回归（LARS）
LARS是一种高效的LASSO求解算法：

```python
from sklearn.linear_model import LassoLars

# 创建模型
lasso_lars = LassoLars(alpha=0.1)

# 训练模型
lasso_lars.fit(X_train, y_train)
```

### 3.3 LASSO的性质

#### 3.3.1 特征选择
LASSO可以自动进行特征选择：

```python
# 查看非零系数
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"选择的特征数量: {len(selected_features)}")
print(f"选择的特征: {selected_features}")
```

#### 3.3.2 软阈值
LASSO的解可以看作是对OLS估计的软阈值：

$$ \hat{\beta}_j^{lasso} = \text{sign}(\hat{\beta}_j^{OLS}) \cdot \max(0, |\hat{\beta}_j^{OLS}| - \lambda) $$

### 3.4 LASSO vs 岭回归

| 特性 | LASSO | 岭回归 |
|------|-------|--------|
| 惩罚项 | L1范数 | L2范数 |
| 解的稀疏性 | 稀疏 | 不稀疏 |
| 特征选择 | 自动 | 不能 |
| 多重共线性 | 处理效果差 | 处理效果好 |
| 计算复杂度 | 较高 | 较低 |
| 适用场景 | 特征选择 | 处理多重共线性 |

## 第四章：弹性网络（Elastic Net）

### 4.1 弹性网络的原理

#### 4.1.1 组合惩罚
弹性网络结合了L1和L2正则化：

$$ \min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2 \right] $$

或者写成：

$$ \min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \left( \alpha \sum_{j=1}^{p} |\beta_j| + \frac{1-\alpha}{2} \sum_{j=1}^{p} \beta_j^2 \right) \right] $$

其中：
- $\lambda$ 是正则化强度
- $\alpha$ 是L1和L2的混合比例（0 ≤ α ≤ 1）

### 4.2 弹性网络的优点

#### 4.2.1 结合优点
- **L1的优点**：特征选择、稀疏性
- **L2的优点**：处理多重共线性、稳定估计

#### 4.2.2 处理高度相关特征
当特征高度相关时，LASSO倾向于随机选择其中一个，而弹性网络会选择多个：

```python
from sklearn.linear_model import ElasticNet

# 创建模型
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

# 训练模型
elastic_net.fit(X_train, y_train)
```

### 4.3 参数调优

#### 4.3.1 网格搜索
使用网格搜索选择最优参数：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

# 参数网格
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# 创建模型
elastic_net = ElasticNet(max_iter=10000)

# 网格搜索
grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 最优参数
best_params = grid_search.best_params_
```

## 第五章：正则化路径

### 5.1 正则化路径的概念

正则化路径描述了随着正则化强度变化，回归系数的变化轨迹。

### 5.2 绘制正则化路径

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, ridge_path

# LASSO路径
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=np.logspace(-4, 1, 100))

# 绘制路径
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for coef in coefs_lasso:
    plt.plot(alphas_lasso, coef)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 第六章：实际应用

### 6.1 基因表达数据分析

**数据集**：基因表达数据
**特征数量**：数千个基因
**样本数量**：数百个样本
**挑战**：
- 高维数据
- 特征高度相关
- 需要特征选择

**解决方案**：使用弹性网络

```python
from sklearn.linear_model import ElasticNetCV

# 创建模型
elastic_net_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], cv=5)

# 训练模型
elastic_net_cv.fit(X_train, y_train)

# 最优参数
print(f"最优alpha: {elastic_net_cv.alpha_}")
print(f"最优l1_ratio: {elastic_net_cv.l1_ratio_}")

# 选择的基因
selected_genes = np.where(elastic_net_cv.coef_ != 0)[0]
print(f"选择的基因数量: {len(selected_genes)}")
```

### 6.2 房价预测

**数据集**：房价数据
**特征**：房屋的各种属性
**目标**：预测房价

**比较不同正则化方法**：

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'LASSO': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}

# 显示结果
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.4f}, R2={metrics['R2']:.4f}")
```

## 第七章：理论分析

### 7.1 偏差-方差分解

正则化回归通过引入偏差来减小方差：

$$ \text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Noise} $$

- **OLS**：无偏，但方差可能很大
- **正则化**：有偏，但方差显著减小
- **最优**：在偏差和方差之间找到平衡

### 7.2 贝叶斯解释

#### 7.2.1 岭回归的贝叶斯解释
岭回归等价于在系数上施加高斯先验：

$$ \beta_j \sim N(0, \tau^2) $$

其中 $\lambda = \frac{\sigma^2}{\tau^2}$。

#### 7.2.2 LASSO的贝叶斯解释
LASSO等价于在系数上施加拉普拉斯先验：

$$ p(\beta_j) = \frac{\lambda}{2} e^{-\lambda|\beta_j|} $$

### 7.3 Oracle性质

在适当条件下，LASSO具有Oracle性质：
- **一致性**：能够正确识别非零系数
- **渐近正态性**：估计量渐近服从正态分布
- **效率**：达到Oracle估计的效率

## 第八章：高级主题

### 8.1 自适应LASSO

自适应LASSO使用不同的权重：

$$ \min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} w_j |\beta_j| \right] $$

其中 $w_j = \frac{1}{|\hat{\beta}_j^{OLS}|^\gamma}$。

### 8.2 组LASSO

当特征可以分成组时，使用组LASSO：

$$ \min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{g=1}^{G} \|\boldsymbol{\beta}_g\|_2 \right] $$

### 8.3 稀疏组LASSO

结合稀疏性和组结构：

$$ \min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{g=1}^{G} \|\boldsymbol{\beta}_g\|_2 \right] $$

## 第九章：常见问题

### 9.1 如何选择正则化方法？

| 场景 | 推荐方法 |
|------|----------|
| 特征选择 | LASSO |
| 多重共线性 | 岭回归 |
| 高维数据 | 弹性网络 |
| 特征分组 | 组LASSO |
| 需要稀疏性和稳定性 | 弹性网络 |

### 9.2 如何调优正则化参数？

1. **交叉验证**：最常用的方法
2. **信息准则**：AIC、BIC
3. **稳定性选择**：基于多次重采样
4. **理论指导**：根据理论结果选择

### 9.3 正则化强度如何选择？

- **太小**：正则化效果不明显，可能过拟合
- **太大**：过度正则化，可能欠拟合
- **适中**：在验证集上表现最好

## 第十章：总结

### 10.1 核心要点

1. **正则化**通过添加惩罚项限制模型复杂度
2. **岭回归**使用L2正则化，处理多重共线性
3. **LASSO**使用L1正则化，进行特征选择
4. **弹性网络**结合L1和L2，兼具两者优点
5. **交叉验证**是选择正则化参数的标准方法

### 10.2 学习路径

1. **基础阶段**：理解正则化的基本原理
2. **实践阶段**：使用Python实现正则化回归
3. **进阶阶段**：学习高级正则化方法
4. **应用阶段**：在实际项目中应用正则化回归

### 10.3 进一步学习

- **贝叶斯回归**：从贝叶斯角度理解正则化
- **稀疏学习**：更广泛的稀疏学习方法
- **在线学习**：在线正则化学习
- **分布式计算**：大规模数据的正则化回归

---

**练习题目**：

1. 证明岭回归估计量是有偏的。
2. 解释为什么LASSO会产生稀疏解。
3. 比较岭回归和LASSO在几何上的差异。
4. 设计一个实验比较不同正则化方法的效果。
5. 实现一个自适应LASSO算法。

**参考资料**：
- 《统计学习方法》李航
- 《机器学习》周志华
- 《The Elements of Statistical Learning》Hastie, Tibshirani, Friedman
- 《Statistical Learning with Sparsity》Hastie, Tibshirani, Wainwright
- 《Regularization Paths for Generalized Linear Models via Coordinate Descent》Friedman et al.
