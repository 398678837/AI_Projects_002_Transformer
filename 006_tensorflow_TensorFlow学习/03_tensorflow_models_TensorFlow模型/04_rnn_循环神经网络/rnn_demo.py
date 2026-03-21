import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("TensorFlow RNN 循环神经网络演示")
print("=" * 50)

# 1. 基本 RNN 模型创建
def create_basic_rnn():
    print("\n1. 创建基本 RNN 模型:")
    
    # 创建 RNN 模型
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(10, 1)),
        tf.keras.layers.Dense(1)
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 2. 编译和训练 RNN 模型
def compile_and_train_rnn():
    print("\n2. 编译和训练 RNN 模型:")
    
    # 创建模型
    model = create_basic_rnn()
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # 生成正弦波数据
    def generate_sine_wave(seq_length, num_samples):
        X = []
        y = []
        for i in range(num_samples):
            start = np.random.rand() * 2 * np.pi
            x = np.linspace(start, start + 2 * np.pi, seq_length + 1)
            X.append(np.sin(x[:-1]).reshape(-1, 1))
            y.append(np.sin(x[-1]))
        return np.array(X), np.array(y)
    
    # 生成训练数据
    seq_length = 10
    X_train, y_train = generate_sine_wave(seq_length, 1000)
    X_test, y_test = generate_sine_wave(seq_length, 100)
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1
    )
    
    # 评估模型
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"测试 MAE: {test_mae:.4f}")
    
    return model, history, X_test, y_test

# 3. 使用 LSTM 模型
def create_lstm_model():
    print("\n3. 创建 LSTM 模型:")
    
    # 创建 LSTM 模型
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(10, 1)),
        tf.keras.layers.Dense(1)
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 4. 使用 GRU 模型
def create_gru_model():
    print("\n4. 创建 GRU 模型:")
    
    # 创建 GRU 模型
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, activation='relu', input_shape=(10, 1)),
        tf.keras.layers.Dense(1)
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 5. 多层 RNN 模型
def create_multi_layer_rnn():
    print("\n5. 创建多层 RNN 模型:")
    
    # 创建多层 LSTM 模型
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(10, 1)),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 6. 双向 RNN 模型
def create_bidirectional_rnn():
    print("\n6. 创建双向 RNN 模型:")
    
    # 创建双向 LSTM 模型
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu'), input_shape=(10, 1)),
        tf.keras.layers.Dense(1)
    ])
    
    # 打印模型结构
    model.summary()
    
    return model

# 7. 文本分类任务
def text_classification():
    print("\n7. 文本分类任务:")
    
    # 加载 IMDB 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    
    # 数据预处理
    max_length = 100
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
    
    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
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

# 8. 时间序列预测
def time_series_prediction():
    print("\n8. 时间序列预测:")
    
    # 生成时间序列数据
    def generate_time_series(n_samples, seq_length):
        X = []
        y = []
        for i in range(n_samples):
            # 生成带有趋势和噪声的时间序列
            t = np.arange(seq_length + 1)
            trend = 0.1 * t
            seasonality = np.sin(0.1 * t) * 10
            noise = np.random.randn(seq_length + 1) * 0.5
            series = trend + seasonality + noise
            X.append(series[:-1].reshape(-1, 1))
            y.append(series[-1])
        return np.array(X), np.array(y)
    
    # 生成数据
    seq_length = 20
    X_train, y_train = generate_time_series(1000, seq_length)
    X_test, y_test = generate_time_series(100, seq_length)
    
    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(seq_length, 1)),
        tf.keras.layers.Dense(1)
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1
    )
    
    # 评估模型
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"测试 MAE: {test_mae:.4f}")
    
    # 可视化预测结果
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='真实值')
    plt.plot(predictions, label='预测值')
    plt.legend()
    plt.title('时间序列预测结果')
    plt.savefig(os.path.join(images_dir, 'time_series_prediction.png'))
    print("时间序列预测结果已保存到 images/time_series_prediction.png")
    
    return model, history

# 9. 模型保存和加载
def save_and_load_rnn():
    print("\n9. 模型保存和加载:")
    
    # 创建并训练模型
    model, _, _, _ = compile_and_train_rnn()
    
    # 保存模型
    model_path = os.path.join(script_dir, 'rnn_model')
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载模型
    loaded_model = tf.keras.models.load_model(model_path)
    print("模型已加载")
    
    # 验证加载的模型
    loaded_model.summary()
    
    return loaded_model

# 10. 超参数调优
def hyperparameter_tuning():
    print("\n10. 超参数调优:")
    
    # 生成数据
    def generate_sine_wave(seq_length, num_samples):
        X = []
        y = []
        for i in range(num_samples):
            start = np.random.rand() * 2 * np.pi
            x = np.linspace(start, start + 2 * np.pi, seq_length + 1)
            X.append(np.sin(x[:-1]).reshape(-1, 1))
            y.append(np.sin(x[-1]))
        return np.array(X), np.array(y)
    
    seq_length = 10
    X_train, y_train = generate_sine_wave(seq_length, 1000)
    X_val, y_val = generate_sine_wave(seq_length, 100)
    
    # 尝试不同的隐藏单元数量
    hidden_units_list = [32, 64, 128]
    results = []
    
    for hidden_units in hidden_units_list:
        print(f"\n训练隐藏单元数为 {hidden_units} 的模型:")
        
        # 创建模型
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_units, activation='relu', input_shape=(seq_length, 1)),
            tf.keras.layers.Dense(1)
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        # 评估模型
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        results.append((hidden_units, val_mae))
        print(f"验证 MAE: {val_mae:.4f}")
    
    # 打印最佳结果
    best_hidden_units, best_mae = min(results, key=lambda x: x[1])
    print(f"\n最佳隐藏单元数: {best_hidden_units}, 最佳 MAE: {best_mae:.4f}")
    
    return results

if __name__ == "__main__":
    # 执行所有演示
    create_basic_rnn()
    compile_and_train_rnn()
    create_lstm_model()
    create_gru_model()
    create_multi_layer_rnn()
    create_bidirectional_rnn()
    text_classification()
    time_series_prediction()
    save_and_load_rnn()
    hyperparameter_tuning()
    
    print("\n" + "=" * 50)
    print("演示完成！")