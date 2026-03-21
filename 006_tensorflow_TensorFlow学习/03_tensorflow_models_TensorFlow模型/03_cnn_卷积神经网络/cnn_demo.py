import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("TensorFlow CNN 卷积神经网络演示")
print("=" * 50)

# 1. 基本 CNN 模型创建
def create_basic_cnn():
    print("\n1. 创建基本 CNN 模型:")
    
    # 创建 CNN 模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 2. 编译和训练 CNN 模型
def compile_and_train_cnn():
    print("\n2. 编译和训练 CNN 模型:")
    
    # 创建模型
    model = create_basic_cnn()
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
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

# 3. 可视化卷积层的输出
def visualize_conv_outputs():
    print("\n3. 可视化卷积层的输出:")
    
    # 创建模型
    model = create_basic_cnn()
    
    # 加载一张测试图像
    (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    img = x_test[0].reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # 创建一个新模型，输出前几个卷积层的结果
    layer_outputs = [layer.output for layer in model.layers[:4]]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # 获取激活值
    activations = activation_model.predict(img)
    
    # 可视化第一个卷积层的输出
    first_layer_activation = activations[0]
    print(f"第一个卷积层的输出形状: {first_layer_activation.shape}")
    
    # 保存可视化结果
    plt.figure(figsize=(10, 10))
    for i in range(32):
        plt.subplot(6, 6, i+1)
        plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'conv_visualization.png'))
    print("卷积层输出可视化已保存到 images/conv_visualization.png")

# 4. 使用不同的池化层
def use_different_pooling():
    print("\n4. 使用不同的池化层:")
    
    # 创建使用最大池化的模型
    model_max_pool = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 创建使用平均池化的模型
    model_avg_pool = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    print("使用最大池化的模型:")
    model_max_pool.summary()
    
    print("\n使用平均池化的模型:")
    model_avg_pool.summary()
    
    return model_max_pool, model_avg_pool

# 5. 数据增强
def use_data_augmentation():
    print("\n5. 使用数据增强:")
    
    # 创建数据增强层
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])
    
    # 创建包含数据增强的模型
    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    print("包含数据增强的模型:")
    model.summary()
    
    return model

# 6. 深层 CNN 模型
def create_deep_cnn():
    print("\n6. 创建深层 CNN 模型:")
    
    # 创建深层 CNN 模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    print("深层 CNN 模型:")
    model.summary()
    
    return model

# 7. 使用预训练模型
def use_pretrained_model():
    print("\n7. 使用预训练模型:")
    
    # 加载预训练的 MobileNetV2 模型
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 冻结基础模型
    base_model.trainable = False
    
    # 创建新模型
    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("使用预训练模型的新模型:")
    model.summary()
    
    return model

# 8. 自定义 CNN 层
def create_custom_cnn():
    print("\n8. 创建自定义 CNN 模型:")
    
    # 使用 Functional API 创建自定义 CNN 模型
    inputs = tf.keras.Input(shape=(28, 28, 1))
    
    # 第一个卷积块
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # 第二个卷积块
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # 第三个卷积块
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # 分类器
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("自定义 CNN 模型:")
    model.summary()
    
    return model

# 9. 模型评估和预测
def evaluate_and_predict():
    print("\n9. 模型评估和预测:")
    
    # 创建并训练模型
    model, _ = compile_and_train_cnn()
    
    # 加载测试数据
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"测试准确率: {test_acc:.4f}")
    
    # 进行预测
    predictions = model.predict(x_test[:5])
    
    # 打印预测结果
    for i in range(5):
        print(f"样本 {i+1}:")
        print(f"预测结果: {np.argmax(predictions[i])}")
        print(f"真实标签: {np.argmax(y_test[i])}")
        print()
    
    return predictions

# 10. 模型保存和加载
def save_and_load_cnn():
    print("\n10. 模型保存和加载:")
    
    # 创建并训练模型
    model, _ = compile_and_train_cnn()
    
    # 保存模型
    model_path = os.path.join(script_dir, 'cnn_model')
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载模型
    loaded_model = tf.keras.models.load_model(model_path)
    print("模型已加载")
    
    # 验证加载的模型
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    test_loss, test_acc = loaded_model.evaluate(x_test, y_test, verbose=2)
    print(f"加载模型的测试准确率: {test_acc:.4f}")
    
    return loaded_model

if __name__ == "__main__":
    # 执行所有演示
    create_basic_cnn()
    compile_and_train_cnn()
    visualize_conv_outputs()
    use_different_pooling()
    use_data_augmentation()
    create_deep_cnn()
    use_pretrained_model()
    create_custom_cnn()
    evaluate_and_predict()
    save_and_load_cnn()
    
    print("\n" + "=" * 50)
    print("演示完成！")