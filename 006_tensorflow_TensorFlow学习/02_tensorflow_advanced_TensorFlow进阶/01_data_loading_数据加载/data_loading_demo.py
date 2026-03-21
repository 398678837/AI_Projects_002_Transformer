import tensorflow as tf
import numpy as np
import os
import time

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("TensorFlow 数据加载演示")
print("=" * 50)

# 1. 基本数据加载
def basic_data_loading():
    print("\n1. 基本数据加载:")
    
    # 从NumPy数组创建数据集
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # 打印数据集信息
    print(f"数据集元素类型: {dataset.element_spec}")
    
    # 遍历数据集
    print("前3个元素:")
    for i, (x_val, y_val) in enumerate(dataset.take(3)):
        print(f"元素 {i}: x.shape={x_val.shape}, y.shape={y_val.shape}")

# 2. 批处理和缓存
def batch_and_cache():
    print("\n2. 批处理和缓存:")
    
    # 创建数据集
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # 批处理
    batch_size = 32
    dataset = dataset.batch(batch_size)
    
    # 缓存
    dataset = dataset.cache()
    
    # 打印批处理后的数据集信息
    print(f"批处理后数据集元素类型: {dataset.element_spec}")
    
    # 遍历批处理后的数据集
    print("前2个批次:")
    for i, (x_batch, y_batch) in enumerate(dataset.take(2)):
        print(f"批次 {i}: x.shape={x_batch.shape}, y.shape={y_batch.shape}")

# 3. 数据增强
def data_augmentation():
    print("\n3. 数据增强:")
    
    # 创建图像数据集
    def load_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = img / 255.0
        return img
    
    # 模拟图像路径
    # 注意：实际使用时需要替换为真实的图像路径
    img_paths = [f"image_{i}.jpg" for i in range(10)]
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    
    # 加载图像
    dataset = dataset.map(load_image)
    
    # 数据增强函数
    def augment(image):
        # 随机翻转
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # 随机旋转
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        # 随机亮度调整
        image = tf.image.random_brightness(image, max_delta=0.1)
        # 随机对比度调整
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image
    
    # 应用数据增强
    dataset = dataset.map(augment)
    
    # 批处理
    dataset = dataset.batch(4)
    
    print("数据增强后的数据集元素类型: {dataset.element_spec}")

# 4. 并行处理
def parallel_processing():
    print("\n4. 并行处理:")
    
    # 创建数据集
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # 定义一个耗时的映射函数
    def time_consuming_map(x, y):
        # 模拟耗时操作
        tf.py_function(lambda: time.sleep(0.01), [], [])
        return x * 2, y * 2
    
    # 串行处理
    start_time = time.time()
    for _ in dataset.map(time_consuming_map).take(100):
        pass
    serial_time = time.time() - start_time
    print(f"串行处理时间: {serial_time:.4f}秒")
    
    # 并行处理
    start_time = time.time()
    for _ in dataset.map(time_consuming_map, num_parallel_calls=tf.data.AUTOTUNE).take(100):
        pass
    parallel_time = time.time() - start_time
    print(f"并行处理时间: {parallel_time:.4f}秒")
    print(f"性能提升: {serial_time/parallel_time:.2f}倍")

# 5. 从文件加载数据
def load_from_files():
    print("\n5. 从文件加载数据:")
    
    # 创建临时文件
    import tempfile
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 创建一些临时文件
    for i in range(10):
        file_path = os.path.join(temp_dir, f"data_{i}.txt")
        with open(file_path, 'w') as f:
            for j in range(10):
                f.write(f"{i},{j},{i*10+j}\n")
    
    # 定义解析函数
    def parse_line(line):
        parts = tf.strings.split(line, ',')
        return tf.strings.to_number(parts[0], out_type=tf.float32), tf.strings.to_number(parts[1], out_type=tf.float32), tf.strings.to_number(parts[2], out_type=tf.float32)
    
    # 创建文件数据集
    file_dataset = tf.data.Dataset.list_files(os.path.join(temp_dir, "*.txt"))
    
    # 读取文件内容
    dataset = file_dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename))
    
    # 解析数据
    dataset = dataset.map(parse_line)
    
    # 批处理
    dataset = dataset.batch(5)
    
    # 打印数据
    print("从文件加载的数据:")
    for i, (x1, x2, y) in enumerate(dataset.take(2)):
        print(f"批次 {i}:")
        print(f"  x1: {x1.numpy()}")
        print(f"  x2: {x2.numpy()}")
        print(f"  y: {y.numpy()}")

# 6. 从TFRecord加载数据
def load_from_tfrecord():
    print("\n6. 从TFRecord加载数据:")
    
    # 创建临时TFRecord文件
    import tempfile
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.tfrecord', delete=False)
    temp_file_path = temp_file.name
    temp_file.close()
    
    # 写入TFRecord文件
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    # 写入数据
    with tf.io.TFRecordWriter(temp_file_path) as writer:
        for i in range(10):
            # 创建特征
            feature = {
                'image': _bytes_feature(tf.io.serialize_tensor(tf.random.normal([32, 32, 3])).numpy()),
                'label': _int64_feature(i % 10)
            }
            # 创建示例
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # 写入示例
            writer.write(example.SerializeToString())
    
    # 解析TFRecord文件
    def parse_tfrecord_fn(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
        label = example['label']
        return image, label
    
    # 创建TFRecord数据集
    dataset = tf.data.TFRecordDataset(temp_file_path)
    
    # 解析数据
    dataset = dataset.map(parse_tfrecord_fn)
    
    # 批处理
    dataset = dataset.batch(3)
    
    # 打印数据
    print("从TFRecord加载的数据:")
    for i, (images, labels) in enumerate(dataset.take(2)):
        print(f"批次 {i}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签: {labels.numpy()}")

# 7. 性能优化
def performance_optimization():
    print("\n7. 性能优化:")
    
    # 创建大型数据集
    x = np.random.random((10000, 100))
    y = np.random.random((10000, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # 优化策略
    dataset = dataset.cache()  # 缓存数据
    dataset = dataset.shuffle(buffer_size=1000)  # 打乱数据
    dataset = dataset.batch(32)  # 批处理
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 预取数据
    
    # 测量性能
    start_time = time.time()
    for _ in dataset.take(100):
        pass
    elapsed_time = time.time() - start_time
    print(f"优化后处理100批次的时间: {elapsed_time:.4f}秒")

# 8. 自定义数据集
def custom_dataset():
    print("\n8. 自定义数据集:")
    
    # 创建自定义数据集
    class CustomDataset(tf.data.Dataset):
        def _generator():
            for i in range(10):
                yield tf.constant(i), tf.constant(i * 2)
        
        def __new__(cls):
            return tf.data.Dataset.from_generator(
                cls._generator,
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            )
    
    # 使用自定义数据集
    dataset = CustomDataset()
    
    # 打印数据
    print("自定义数据集的数据:")
    for x, y in dataset:
        print(f"x: {x.numpy()}, y: {y.numpy()}")

# 9. 数据集管道
def dataset_pipeline():
    print("\n9. 数据集管道:")
    
    # 创建数据集
    def create_dataset():
        # 从NumPy数组创建数据集
        x = np.random.random((1000, 28, 28, 1))
        y = np.random.randint(0, 10, (1000,))
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        
        # 数据增强
        def augment(image, label):
            # 随机翻转
            image = tf.image.random_flip_left_right(image)
            # 随机旋转
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            # 随机亮度调整
            image = tf.image.random_brightness(image, max_delta=0.1)
            return image, label
        
        # 构建管道
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(32)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    # 创建训练和验证数据集
    train_dataset = create_dataset()
    val_dataset = create_dataset()
    
    # 打印数据集信息
    print(f"训练数据集元素类型: {train_dataset.element_spec}")
    print(f"验证数据集元素类型: {val_dataset.element_spec}")

# 10. 与Keras集成
def integrate_with_keras():
    print("\n10. 与Keras集成:")
    
    # 创建数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(60000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 训练模型
    print("开始训练模型...")
    history = model.fit(train_dataset, epochs=2, validation_data=test_dataset)
    
    # 评估模型
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"测试准确率: {test_acc:.4f}")

if __name__ == "__main__":
    # 执行所有演示
    basic_data_loading()
    batch_and_cache()
    data_augmentation()
    parallel_processing()
    load_from_files()
    load_from_tfrecord()
    performance_optimization()
    custom_dataset()
    dataset_pipeline()
    integrate_with_keras()
    
    print("\n" + "=" * 50)
    print("演示完成！")