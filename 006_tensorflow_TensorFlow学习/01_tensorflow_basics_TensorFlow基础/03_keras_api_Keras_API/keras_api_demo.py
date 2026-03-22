#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow Keras API 演示

本脚本演示 TensorFlow Keras API 的使用方法和基本操作。
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("TensorFlow Keras API 演示")
print("=" * 50)

# 1. 数据集准备
def prepare_dataset():
    print("\n1. 数据集准备:")
    
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # 标签独热编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")
    print(f"训练标签形状: {y_train.shape}")
    print(f"测试标签形状: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

# 2. Sequential 模型
def create_sequential_model():
    print("\n2. Sequential 模型:")
    
    # 创建 Sequential 模型
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 打印模型摘要
    model.summary()
    
    return model

# 3. Functional API 模型
def create_functional_model():
    print("\n3. Functional API 模型:")
    
    # 创建输入层
    inputs = layers.Input(shape=(28, 28, 1))
    
    # 创建中间层
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 打印模型摘要
    model.summary()
    
    return model

# 4. 自定义模型类
def create_custom_model():
    print("\n4. 自定义模型类:")
    
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
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 打印模型摘要
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()
    
    return model

# 5. 模型训练
def train_model(model, x_train, y_train, x_test, y_test):
    print("\n5. 模型训练:")
    
    # 训练模型
    history = model.fit(x_train, y_train,
                      epochs=5,
                      batch_size=32,
                      validation_data=(x_test, y_test))
    
    # 打印训练结果
    print(f"训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    
    return history

# 6. 模型评估
def evaluate_model(model, x_test, y_test):
    print("\n6. 模型评估:")
    
    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"测试损失: {loss:.4f}")
    print(f"测试准确率: {accuracy:.4f}")

# 7. 模型预测
def predict_with_model(model, x_test):
    print("\n7. 模型预测:")
    
    # 预测
    predictions = model.predict(x_test[:5])
    
    # 打印预测结果
    for i in range(5):
        predicted_class = np.argmax(predictions[i])
        print(f"样本 {i+1} 预测类别: {predicted_class}")

# 8. 模型保存和加载
def save_and_load_model(model):
    print("\n8. 模型保存和加载:")
    
    # 保存模型
    model.save('mnist_model.h5')
    print("模型已保存为 mnist_model.h5")
    
    # 加载模型
    loaded_model = tf.keras.models.load_model('mnist_model.h5')
    print("模型已加载")
    
    return loaded_model

# 9. 回调函数
def use_callbacks():
    print("\n9. 回调函数:")
    
    # 创建回调函数
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
    
    print("已创建回调函数: EarlyStopping, ReduceLROnPlateau, TensorBoard")
    
    return callbacks

# 10. 数据增强
def data_augmentation():
    print("\n10. 数据增强:")
    
    # 创建数据增强层
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])
    
    print("已创建数据增强层")
    
    return data_augmentation

if __name__ == "__main__":
    # 准备数据集
    (x_train, y_train), (x_test, y_test) = prepare_dataset()
    
    # 创建并训练 Sequential 模型
    print("\n" + "=" * 30)
    print("使用 Sequential 模型")
    print("=" * 30)
    sequential_model = create_sequential_model()
    train_model(sequential_model, x_train, y_train, x_test, y_test)
    evaluate_model(sequential_model, x_test, y_test)
    predict_with_model(sequential_model, x_test)
    
    # 创建并训练 Functional API 模型
    print("\n" + "=" * 30)
    print("使用 Functional API 模型")
    print("=" * 30)
    functional_model = create_functional_model()
    train_model(functional_model, x_train, y_train, x_test, y_test)
    evaluate_model(functional_model, x_test, y_test)
    predict_with_model(functional_model, x_test)
    
    # 创建并训练自定义模型
    print("\n" + "=" * 30)
    print("使用自定义模型类")
    print("=" * 30)
    custom_model = create_custom_model()
    train_model(custom_model, x_train, y_train, x_test, y_test)
    evaluate_model(custom_model, x_test, y_test)
    predict_with_model(custom_model, x_test)
    
    # 模型保存和加载
    print("\n" + "=" * 30)
    print("模型保存和加载")
    print("=" * 30)
    loaded_model = save_and_load_model(sequential_model)
    evaluate_model(loaded_model, x_test, y_test)
    
    # 回调函数和数据增强
    print("\n" + "=" * 30)
    print("回调函数和数据增强")
    print("=" * 30)
    callbacks = use_callbacks()
    augmentation = data_augmentation()
    
    print("\n" + "=" * 50)
    print("演示完成！")