import tensorflow as tf
import numpy as np
import os
import time

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("TensorFlow 分布式训练演示")
print("=" * 50)

# 1. 检查可用设备
def check_devices():
    print("\n1. 检查可用设备:")
    
    # 获取可用的GPU设备
    gpus = tf.config.list_physical_devices('GPU')
    print(f"可用的GPU数量: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.name}")
    
    # 获取可用的CPU设备
    cpus = tf.config.list_physical_devices('CPU')
    print(f"可用的CPU数量: {len(cpus)}")
    for i, cpu in enumerate(cpus):
        print(f"CPU {i}: {cpu.name}")

# 2. 单设备训练（基准）
def single_device_training():
    print("\n2. 单设备训练（基准）:")
    
    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 生成训练数据
    x_train = tf.random.normal((10000, 1000))
    y_train = tf.one_hot(tf.random.uniform((10000,), maxval=10, dtype=tf.int32), depth=10)
    
    # 训练模型
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=32)
    end_time = time.time()
    
    print(f"单设备训练时间: {end_time - start_time:.2f}秒")
    print(f"最终损失: {history.history['loss'][-1]:.4f}")
    print(f"最终准确率: {history.history['accuracy'][-1]:.4f}")

# 3. 数据并行训练
def data_parallel_training():
    print("\n3. 数据并行训练:")
    
    # 检查是否有多个GPU
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2:
        print("需要至少2个GPU进行数据并行训练")
        return
    
    # 创建分布式策略
    strategy = tf.distribute.MirroredStrategy()
    print(f"使用的设备: {strategy.num_replicas_in_sync}")
    
    # 在策略范围内创建模型
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # 生成训练数据
    x_train = tf.random.normal((10000, 1000))
    y_train = tf.one_hot(tf.random.uniform((10000,), maxval=10, dtype=tf.int32), depth=10)
    
    # 训练模型
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=32 * strategy.num_replicas_in_sync)
    end_time = time.time()
    
    print(f"数据并行训练时间: {end_time - start_time:.2f}秒")
    print(f"最终损失: {history.history['loss'][-1]:.4f}")
    print(f"最终准确率: {history.history['accuracy'][-1]:.4f}")

# 4. 自定义训练循环的分布式训练
def custom_distributed_training():
    print("\n4. 自定义训练循环的分布式训练:")
    
    # 检查是否有多个GPU
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2:
        print("需要至少2个GPU进行分布式训练")
        return
    
    # 创建分布式策略
    strategy = tf.distribute.MirroredStrategy()
    print(f"使用的设备: {strategy.num_replicas_in_sync}")
    
    # 生成训练数据
    x_train = tf.random.normal((10000, 1000))
    y_train = tf.one_hot(tf.random.uniform((10000,), maxval=10, dtype=tf.int32), depth=10)
    
    # 创建分布式数据集
    batch_size = 32
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).batch(global_batch_size)
    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    
    # 在策略范围内创建模型和优化器
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        optimizer = tf.optimizers.Adam()
        loss_fn = tf.losses.CategoricalCrossentropy()
    
    # 自定义训练步骤
    @tf.function
    def train_step(inputs):
        x, y = inputs
        
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    # 分布式训练步骤
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    # 训练模型
    epochs = 5
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in distributed_dataset:
            loss = distributed_train_step(batch)
            total_loss += loss
            num_batches += 1
        
        average_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
    
    end_time = time.time()
    print(f"自定义分布式训练时间: {end_time - start_time:.2f}秒")

# 5. 多工作进程分布式训练
def multi_worker_training():
    print("\n5. 多工作进程分布式训练:")
    print("注意: 此演示需要在多台机器上运行，这里仅展示代码结构")
    
    # 设置环境变量
    os.environ['TF_CONFIG'] = '''
    {
      "cluster": {
        "worker": [
          "localhost:12345",
          "localhost:23456"
        ]
      },
      "task": {
        "type": "worker",
        "index": 0
      }
    }
    '''
    
    # 创建分布式策略
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"使用的策略: {strategy}")
    
    # 在策略范围内创建模型
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    print("多工作进程分布式训练模型已创建")

# 6. 参数服务器分布式训练
def parameter_server_training():
    print("\n6. 参数服务器分布式训练:")
    print("注意: 此演示需要设置参数服务器，这里仅展示代码结构")
    
    # 设置环境变量
    os.environ['TF_CONFIG'] = '''
    {
      "cluster": {
        "ps": ["localhost:12345"],
        "worker": ["localhost:23456", "localhost:34567"]
      },
      "task": {
        "type": "worker",
        "index": 0
      }
    }
    '''
    
    # 创建分布式策略
    strategy = tf.distribute.ParameterServerStrategy()
    print(f"使用的策略: {strategy}")
    
    # 在策略范围内创建模型
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    print("参数服务器分布式训练模型已创建")

# 7. 混合精度训练
def mixed_precision_training():
    print("\n7. 混合精度训练:")
    
    # 启用混合精度
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"混合精度策略: {tf.keras.mixed_precision.global_policy()}")
    
    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 生成训练数据
    x_train = tf.random.normal((10000, 1000))
    y_train = tf.one_hot(tf.random.uniform((10000,), maxval=10, dtype=tf.int32), depth=10)
    
    # 训练模型
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=32)
    end_time = time.time()
    
    print(f"混合精度训练时间: {end_time - start_time:.2f}秒")
    print(f"最终损失: {history.history['loss'][-1]:.4f}")
    print(f"最终准确率: {history.history['accuracy'][-1]:.4f}")

# 8. 分布式训练的性能优化
def distributed_performance_optimization():
    print("\n8. 分布式训练的性能优化:")
    
    # 检查是否有多个GPU
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2:
        print("需要至少2个GPU进行分布式训练")
        return
    
    # 创建分布式策略
    strategy = tf.distribute.MirroredStrategy()
    print(f"使用的设备: {strategy.num_replicas_in_sync}")
    
    # 在策略范围内创建模型
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # 生成训练数据
    x_train = tf.random.normal((10000, 1000))
    y_train = tf.one_hot(tf.random.uniform((10000,), maxval=10, dtype=tf.int32), depth=10)
    
    # 优化数据集
    batch_size = 32
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # 训练模型
    start_time = time.time()
    history = model.fit(dataset, epochs=5)
    end_time = time.time()
    
    print(f"优化后的分布式训练时间: {end_time - start_time:.2f}秒")
    print(f"最终损失: {history.history['loss'][-1]:.4f}")
    print(f"最终准确率: {history.history['accuracy'][-1]:.4f}")

if __name__ == "__main__":
    # 执行所有演示
    check_devices()
    single_device_training()
    data_parallel_training()
    custom_distributed_training()
    multi_worker_training()
    parameter_server_training()
    mixed_precision_training()
    distributed_performance_optimization()
    
    print("\n" + "=" * 50)
    print("演示完成！")