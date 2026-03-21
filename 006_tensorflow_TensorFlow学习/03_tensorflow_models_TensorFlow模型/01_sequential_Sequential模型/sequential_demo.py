import tensorflow as tf
import numpy as np
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("TensorFlow Sequential 模型演示")
print("=" * 50)

# 1. 基本 Sequential 模型创建
def create_basic_sequential_model():
    print("\n1. 创建基本 Sequential 模型:")
    
    # 创建一个简单的 Sequential 模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 2. 逐步构建 Sequential 模型
def build_sequential_model_step_by_step():
    print("\n2. 逐步构建 Sequential 模型:")
    
    # 创建一个空的 Sequential 模型
    model = tf.keras.Sequential()
    
    # 逐步添加层
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    # 打印模型结构
    model.summary()
    
    return model

# 3. 编译和训练模型
def compile_and_train_model():
    print("\n3. 编译和训练模型:")
    
    # 创建模型
    model = create_basic_sequential_model()
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"测试准确率: {test_acc:.4f}")
    
    return model, history

# 4. 使用不同的激活函数
def use_different_activations():
    print("\n4. 使用不同的激活函数:")
    
    # 创建使用不同激活函数的模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 5. 添加 Dropout 层
def add_dropout_layers():
    print("\n5. 添加 Dropout 层:")
    
    # 创建带有 Dropout 层的模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 6. 使用批量归一化
def use_batch_normalization():
    print("\n6. 使用批量归一化:")
    
    # 创建带有批量归一化的模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(784,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 7. 保存和加载模型
def save_and_load_model():
    print("\n7. 保存和加载模型:")
    
    # 创建并训练模型
    model, _ = compile_and_train_model()
    
    # 保存模型
    model_path = os.path.join(script_dir, 'saved_model')
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载模型
    loaded_model = tf.keras.models.load_model(model_path)
    print("模型已加载")
    
    # 验证加载的模型
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    test_loss, test_acc = loaded_model.evaluate(x_test, y_test, verbose=2)
    print(f"加载模型的测试准确率: {test_acc:.4f}")
    
    return loaded_model

# 8. 使用回调函数
def use_callbacks():
    print("\n8. 使用回调函数:")
    
    # 创建模型
    model = create_basic_sequential_model()
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(script_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(script_dir, 'logs'))
    ]
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks
    )
    
    return model, history

# 9. 自定义损失函数
def use_custom_loss():
    print("\n9. 使用自定义损失函数:")
    
    # 定义自定义损失函数
    def custom_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 创建模型
    model = create_basic_sequential_model()
    
    # 编译模型，使用自定义损失函数
    model.compile(
        optimizer='adam',
        loss=custom_loss,
        metrics=['accuracy']
    )
    
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=32,
        validation_split=0.1
    )
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"测试准确率: {test_acc:.4f}")
    
    return model

# 10. 模型预测
def model_prediction():
    print("\n10. 模型预测:")
    
    # 创建并训练模型
    model, _ = compile_and_train_model()
    
    # 加载测试数据
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # 进行预测
    predictions = model.predict(x_test[:5])
    
    # 打印预测结果
    for i in range(5):
        print(f"样本 {i+1}:")
        print(f"预测结果: {np.argmax(predictions[i])}")
        print(f"真实标签: {y_test[i]}")
        print()
    
    return predictions

if __name__ == "__main__":
    # 执行所有演示
    create_basic_sequential_model()
    build_sequential_model_step_by_step()
    compile_and_train_model()
    use_different_activations()
    add_dropout_layers()
    use_batch_normalization()
    save_and_load_model()
    use_callbacks()
    use_custom_loss()
    model_prediction()
    
    print("\n" + "=" * 50)
    print("演示完成！")