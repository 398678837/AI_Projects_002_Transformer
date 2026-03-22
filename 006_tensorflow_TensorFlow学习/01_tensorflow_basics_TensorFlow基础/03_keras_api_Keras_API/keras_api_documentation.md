# TensorFlow Keras API 详细文档

## 1. Keras 简介

Keras 是一个高级神经网络 API，它是 TensorFlow 的官方高级 API。Keras 使得构建和训练神经网络变得更加简单和直观，同时保持了 TensorFlow 的灵活性和性能。

## 2. 模型类型

### 2.1 Sequential 模型

Sequential 模型是最基本的模型类型，它是一个线性堆叠的层序列。

```python
from tensorflow.keras import models, layers

# 创建 Sequential 模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
```

### 2.2 Functional API 模型

Functional API 允许创建更复杂的模型，如多输入多输出模型、共享层模型等。

```python
from tensorflow.keras import models, layers

# 创建输入层
inputs = layers.Input(shape=(28, 28, 1))

# 创建中间层
x = layers.Flatten()(inputs)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

# 创建模型
model = models.Model(inputs=inputs, outputs=outputs)
```

### 2.3 自定义模型类

通过继承 `tf.keras.Model` 类，可以创建完全自定义的模型。

```python
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

# 创建模型实例
model = CustomModel()
```

## 3. 模型编译

在训练模型之前，需要编译模型，指定优化器、损失函数和评估指标。

```python
model.compile(
    optimizer='adam',  # 优化器
    loss='categorical_crossentropy',  # 损失函数
    metrics=['accuracy']  # 评估指标
)
```

### 3.1 优化器

常用的优化器包括：
- `adam`：自适应矩估计
- `sgd`：随机梯度下降
- `rmsprop`：均方根传播
- `adagrad`：自适应梯度算法
- `adadelta`：自适应学习率

### 3.2 损失函数

常用的损失函数包括：
- `categorical_crossentropy`：多分类交叉熵
- `binary_crossentropy`：二分类交叉熵
- `mean_squared_error`：均方误差
- `mean_absolute_error`：平均绝对误差
- `sparse_categorical_crossentropy`：稀疏多分类交叉熵

### 3.3 评估指标

常用的评估指标包括：
- `accuracy`：准确率
- `precision`：精确率
- `recall`：召回率
- `f1_score`：F1 分数
- `auc`：AUC 分数

## 4. 模型训练

使用 `fit` 方法训练模型。

```python
history = model.fit(
    x_train, y_train,  # 训练数据和标签
    epochs=5,  # 训练轮数
    batch_size=32,  # 批次大小
    validation_data=(x_test, y_test)  # 验证数据
)
```

### 4.1 回调函数

回调函数可以在训练过程中执行特定操作，如早停、学习率调度等。

```python
callbacks = [
    # 早停
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    # 学习率调度
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        min_lr=0.0001
    ),
    # TensorBoard
    tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1
    )
]

# 在训练中使用回调函数
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)
```

## 5. 模型评估

使用 `evaluate` 方法评估模型性能。

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"测试损失: {loss:.4f}")
print(f"测试准确率: {accuracy:.4f}")
```

## 6. 模型预测

使用 `predict` 方法进行预测。

```python
predictions = model.predict(x_test)

# 获取预测类别
predicted_classes = tf.argmax(predictions, axis=1)
```

## 7. 模型保存和加载

### 7.1 保存模型

```python
# 保存整个模型
model.save('model.h5')

# 保存模型权重
model.save_weights('model_weights.h5')
```

### 7.2 加载模型

```python
# 加载整个模型
loaded_model = tf.keras.models.load_model('model.h5')

# 加载模型权重
model = create_model()  # 创建模型架构
model.load_weights('model_weights.h5')
```

## 8. 数据增强

数据增强可以增加训练数据的多样性，提高模型的泛化能力。

```python
from tensorflow.keras import layers

# 创建数据增强层
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# 在模型中使用数据增强
inputs = layers.Input(shape=(28, 28, 1))
x = data_augmentation(inputs)
x = layers.Flatten()(x)
# 其他层...
```

## 9. 常用层

### 9.1 核心层

- `Dense`：全连接层
- `Flatten`：展平层
- `Dropout`： dropout 层
- `BatchNormalization`：批归一化层
- `Reshape`：重塑层

### 9.2 卷积层

- `Conv2D`：二维卷积层
- `MaxPooling2D`：二维最大池化层
- `AveragePooling2D`：二维平均池化层
- `Conv1D`：一维卷积层
- `MaxPooling1D`：一维最大池化层

### 9.3 循环层

- `LSTM`：长短期记忆网络
- `GRU`：门控循环单元
- `SimpleRNN`：简单循环神经网络

### 9.4 激活函数

- `relu`：修正线性单元
- `sigmoid`： sigmoid 函数
- `tanh`：双曲正切函数
- `softmax`： softmax 函数
- `leaky_relu`：带泄漏的修正线性单元

## 10. 常见问题和解决方案

### 10.1 过拟合
**问题**：模型在训练集上表现良好，但在测试集上表现不佳。
**解决方案**：
- 使用 dropout 层
- 使用批归一化
- 增加数据增强
- 减少模型复杂度
- 使用早停

### 10.2 欠拟合
**问题**：模型在训练集和测试集上表现都不佳。
**解决方案**：
- 增加模型复杂度
- 增加训练轮数
- 调整优化器和学习率
- 检查数据预处理

### 10.3 训练速度慢
**问题**：模型训练速度慢。
**解决方案**：
- 使用 GPU 训练
- 增加批次大小
- 使用更高效的优化器（如 Adam）
- 减少模型复杂度

### 10.4 内存不足
**问题**：训练时内存不足。
**解决方案**：
- 减少批次大小
- 减少模型复杂度
- 使用数据生成器
- 清理内存

## 11. 最佳实践

- 从小模型开始，逐渐增加复杂度。
- 使用数据增强提高模型泛化能力。
- 合理设置学习率，使用学习率调度。
- 使用早停避免过拟合。
- 保存模型的最佳权重。
- 使用 TensorBoard 监控训练过程。
- 对模型进行超参数调优。
- 考虑使用预训练模型进行迁移学习。