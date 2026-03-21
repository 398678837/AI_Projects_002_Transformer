import tensorflow as tf
import numpy as np
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("TensorFlow Functional API 演示")
print("=" * 50)

# 1. 基本 Functional API 模型创建
def create_basic_functional_model():
    print("\n1. 创建基本 Functional API 模型:")
    
    # 创建输入层
    inputs = tf.keras.Input(shape=(784,))
    
    # 创建隐藏层
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # 创建输出层
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 打印模型结构
    model.summary()
    
    return model

# 2. 多输入模型
def create_multi_input_model():
    print("\n2. 创建多输入模型:")
    
    # 创建第一个输入（图像）
    image_input = tf.keras.Input(shape=(28, 28, 1), name='image')
    x1 = tf.keras.layers.Flatten()(image_input)
    x1 = tf.keras.layers.Dense(64, activation='relu')(x1)
    
    # 创建第二个输入（额外特征）
    feature_input = tf.keras.Input(shape=(10,), name='feature')
    x2 = tf.keras.layers.Dense(64, activation='relu')(feature_input)
    
    # 合并两个输入
    merged = tf.keras.layers.concatenate([x1, x2])
    
    # 创建输出层
    outputs = tf.keras.layers.Dense(10, activation='softmax')(merged)
    
    # 创建模型
    model = tf.keras.Model(inputs=[image_input, feature_input], outputs=outputs)
    
    # 打印模型结构
    model.summary()
    
    return model

# 3. 多输出模型
def create_multi_output_model():
    print("\n3. 创建多输出模型:")
    
    # 创建输入层
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # 创建多个输出层
    output1 = tf.keras.layers.Dense(10, activation='softmax', name='digit')(x)
    output2 = tf.keras.layers.Dense(1, activation='sigmoid', name='is_odd')(x)
    
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
    
    # 打印模型结构
    model.summary()
    
    return model

# 4. 共享层模型
def create_shared_layer_model():
    print("\n4. 创建共享层模型:")
    
    # 创建输入层
    input_a = tf.keras.Input(shape=(10,))
    input_b = tf.keras.Input(shape=(10,))
    
    # 创建共享层
    shared_layer = tf.keras.layers.Dense(64, activation='relu')
    
    # 应用共享层到两个输入
    x1 = shared_layer(input_a)
    x2 = shared_layer(input_b)
    
    # 创建输出层
    output_a = tf.keras.layers.Dense(1, activation='sigmoid')(x1)
    output_b = tf.keras.layers.Dense(1, activation='sigmoid')(x2)
    
    # 创建模型
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=[output_a, output_b])
    
    # 打印模型结构
    model.summary()
    
    return model

# 5. 复杂网络结构（带有分支）
def create_complex_model():
    print("\n5. 创建复杂网络结构:")
    
    # 创建输入层
    inputs = tf.keras.Input(shape=(28, 28, 1))
    
    # 主干网络
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # 分支1
    branch1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    branch1 = tf.keras.layers.Flatten()(branch1)
    
    # 分支2
    branch2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')(x)
    branch2 = tf.keras.layers.Flatten()(branch2)
    
    # 合并分支
    merged = tf.keras.layers.concatenate([branch1, branch2])
    
    # 创建输出层
    outputs = tf.keras.layers.Dense(10, activation='softmax')(merged)
    
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 打印模型结构
    model.summary()
    
    return model

# 6. 编译和训练多输入多输出模型
def compile_and_train_multi_io_model():
    print("\n6. 编译和训练多输入多输出模型:")
    
    # 创建多输出模型
    model = create_multi_output_model()
    
    # 编译模型，为不同输出指定不同损失函数
    model.compile(
        optimizer='adam',
        loss={'digit': 'categorical_crossentropy', 'is_odd': 'binary_crossentropy'},
        metrics={'digit': 'accuracy', 'is_odd': 'accuracy'}
    )
    
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # 准备多输出标签
    y_digit = tf.keras.utils.to_categorical(y_train, 10)
    y_is_odd = tf.cast(y_train % 2, dtype='float32')
    
    # 训练模型
    history = model.fit(
        x_train, {'digit': y_digit, 'is_odd': y_is_odd},
        epochs=3,
        batch_size=32,
        validation_split=0.1
    )
    
    # 评估模型
    y_test_digit = tf.keras.utils.to_categorical(y_test, 10)
    y_test_is_odd = tf.cast(y_test % 2, dtype='float32')
    
    test_loss, test_digit_loss, test_is_odd_loss, test_digit_acc, test_is_odd_acc = model.evaluate(
        x_test, {'digit': y_test_digit, 'is_odd': y_test_is_odd}, verbose=2
    )
    
    print(f"测试准确率 - 数字识别: {test_digit_acc:.4f}")
    print(f"测试准确率 - 奇偶判断: {test_is_odd_acc:.4f}")
    
    return model, history

# 7. 模型子类化
def create_subclassed_model():
    print("\n7. 创建模型子类化:")
    
    class CustomModel(tf.keras.Model):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.dense1 = tf.keras.layers.Dense(64, activation='relu')
            self.dense2 = tf.keras.layers.Dense(64, activation='relu')
            self.dense3 = tf.keras.layers.Dense(10, activation='softmax')
        
        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dense2(x)
            return self.dense3(x)
    
    # 创建模型
    model = CustomModel()
    
    # 打印模型结构
    model.build(input_shape=(None, 784))
    model.summary()
    
    return model

# 8. 函数式 API 与 Sequential 模型的对比
def compare_sequential_functional():
    print("\n8. 函数式 API 与 Sequential 模型的对比:")
    
    # 创建 Sequential 模型
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    print("Sequential 模型:")
    sequential_model.summary()
    
    # 创建 Functional API 模型
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("\nFunctional API 模型:")
    functional_model.summary()
    
    return sequential_model, functional_model

# 9. 模型保存和加载
def save_and_load_functional_model():
    print("\n9. 保存和加载 Functional API 模型:")
    
    # 创建模型
    model = create_basic_functional_model()
    
    # 保存模型
    model_path = os.path.join(script_dir, 'functional_model')
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载模型
    loaded_model = tf.keras.models.load_model(model_path)
    print("模型已加载")
    
    # 验证加载的模型
    loaded_model.summary()
    
    return loaded_model

# 10. 使用 Functional API 构建自编码器
def create_autoencoder():
    print("\n10. 使用 Functional API 构建自编码器:")
    
    # 编码器
    inputs = tf.keras.Input(shape=(784,))
    encoded = tf.keras.layers.Dense(128, activation='relu')(inputs)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
    
    # 解码器
    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(784, activation='sigmoid')(decoded)
    
    # 创建自编码器模型
    autoencoder = tf.keras.Model(inputs=inputs, outputs=decoded)
    
    # 创建编码器模型
    encoder = tf.keras.Model(inputs=inputs, outputs=encoded)
    
    # 创建解码器模型
    decoder_input = tf.keras.Input(shape=(32,))
    decoder_layer1 = autoencoder.layers[-3](decoder_input)
    decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
    decoder_layer3 = autoencoder.layers[-1](decoder_layer2)
    decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder_layer3)
    
    # 打印模型结构
    print("自编码器模型:")
    autoencoder.summary()
    
    print("\n编码器模型:")
    encoder.summary()
    
    print("\n解码器模型:")
    decoder.summary()
    
    return autoencoder, encoder, decoder

if __name__ == "__main__":
    # 执行所有演示
    create_basic_functional_model()
    create_multi_input_model()
    create_multi_output_model()
    create_shared_layer_model()
    create_complex_model()
    compile_and_train_multi_io_model()
    create_subclassed_model()
    compare_sequential_functional()
    save_and_load_functional_model()
    create_autoencoder()
    
    print("\n" + "=" * 50)
    print("演示完成！")