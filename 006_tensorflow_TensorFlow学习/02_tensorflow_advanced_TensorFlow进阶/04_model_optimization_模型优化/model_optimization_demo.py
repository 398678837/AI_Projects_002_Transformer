import tensorflow as tf
import numpy as np
import os
import time

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("TensorFlow 模型优化演示")
print("=" * 50)

# 1. 基础模型创建
def create_base_model():
    print("\n1. 创建基础模型:")
    
    # 创建一个简单的卷积神经网络模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("基础模型创建完成")
    return model

# 2. 模型量化
def model_quantization():
    print("\n2. 模型量化:")
    
    # 创建并训练基础模型
    model = create_base_model()
    
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 训练模型
    model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.1)
    
    # 评估基础模型
    base_loss, base_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"基础模型 - 损失: {base_loss:.4f}, 准确率: {base_accuracy:.4f}")
    
    # 保存基础模型
    base_model_path = os.path.join(script_dir, 'base_model')
    model.save(base_model_path)
    
    # 1. 动态范围量化
    print("\n2.1 动态范围量化:")
    converter = tf.lite.TFLiteConverter.from_saved_model(base_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    
    # 保存量化模型
    tflite_quant_model_path = os.path.join(script_dir, 'quantized_model.tflite')
    with open(tflite_quant_model_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    # 评估量化模型
    interpreter = tf.lite.Interpreter(model_path=tflite_quant_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 测试量化模型
    correct = 0
    for i in range(len(x_test)):
        input_data = np.expand_dims(x_test[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output) == np.argmax(y_test[i]):
            correct += 1
    
    quant_accuracy = correct / len(x_test)
    print(f"动态范围量化模型 - 准确率: {quant_accuracy:.4f}")
    
    # 2. 整数量化
    print("\n2.2 整数量化:")
    def representative_data_gen():
        for i in range(100):
            yield [x_train[i:i+1]]
    
    converter = tf.lite.TFLiteConverter.from_saved_model(base_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int_quant_model = converter.convert()
    
    # 保存整数量化模型
    tflite_int_quant_model_path = os.path.join(script_dir, 'int_quantized_model.tflite')
    with open(tflite_int_quant_model_path, 'wb') as f:
        f.write(tflite_int_quant_model)
    
    # 评估整数量化模型
    interpreter = tf.lite.Interpreter(model_path=tflite_int_quant_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 测试整数量化模型
    correct = 0
    for i in range(len(x_test)):
        input_data = np.expand_dims(x_test[i], axis=0)
        # 转换为int8
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = input_data / input_scale + input_zero_point
        input_data = input_data.astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # 转换回浮点数
        output_scale, output_zero_point = output_details[0]['quantization']
        output = (output - output_zero_point) * output_scale
        
        if np.argmax(output) == np.argmax(y_test[i]):
            correct += 1
    
    int_quant_accuracy = correct / len(x_test)
    print(f"整数量化模型 - 准确率: {int_quant_accuracy:.4f}")
    
    # 3. 比较模型大小
    base_model_size = os.path.getsize(os.path.join(base_model_path, 'saved_model.pb'))
    quant_model_size = os.path.getsize(tflite_quant_model_path)
    int_quant_model_size = os.path.getsize(tflite_int_quant_model_path)
    
    print("\n2.3 模型大小比较:")
    print(f"基础模型大小: {base_model_size / 1024:.2f} KB")
    print(f"动态范围量化模型大小: {quant_model_size / 1024:.2f} KB")
    print(f"整数量化模型大小: {int_quant_model_size / 1024:.2f} KB")

# 3. 模型剪枝
def model_pruning():
    print("\n3. 模型剪枝:")
    
    # 安装必要的库
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        print("请安装 tensorflow_model_optimization 库: pip install tensorflow-model-optimization")
        return
    
    # 创建基础模型
    model = create_base_model()
    
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 创建剪枝模型
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    
    # 定义剪枝参数
    batch_size = 32
    epochs = 2
    validation_split = 0.1
    
    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=end_step
        )
    }
    
    # 创建剪枝模型
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    
    # 编译剪枝模型
    model_for_pruning.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练剪枝模型
    model_for_pruning.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=os.path.join(script_dir, 'pruning_logs'))
        ]
    )
    
    # 评估剪枝模型
    pruned_loss, pruned_accuracy = model_for_pruning.evaluate(x_test, y_test, verbose=0)
    print(f"剪枝模型 - 损失: {pruned_loss:.4f}, 准确率: {pruned_accuracy:.4f}")
    
    # 移除剪枝包装器
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    # 保存剪枝模型
    pruned_model_path = os.path.join(script_dir, 'pruned_model')
    model_for_export.save(pruned_model_path)
    
    # 比较模型大小
    base_model = create_base_model()
    base_model.save(os.path.join(script_dir, 'base_model_for_pruning'))
    
    base_model_size = os.path.getsize(os.path.join(script_dir, 'base_model_for_pruning', 'saved_model.pb'))
    pruned_model_size = os.path.getsize(os.path.join(pruned_model_path, 'saved_model.pb'))
    
    print("\n3.1 模型大小比较:")
    print(f"基础模型大小: {base_model_size / 1024:.2f} KB")
    print(f"剪枝模型大小: {pruned_model_size / 1024:.2f} KB")

# 4. 模型蒸馏
def model_distillation():
    print("\n4. 模型蒸馏:")
    
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 创建教师模型（复杂模型）
    print("4.1 创建并训练教师模型:")
    teacher_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    teacher_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    teacher_model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)
    teacher_loss, teacher_accuracy = teacher_model.evaluate(x_test, y_test, verbose=0)
    print(f"教师模型 - 损失: {teacher_loss:.4f}, 准确率: {teacher_accuracy:.4f}")
    
    # 创建学生模型（简单模型）
    print("\n4.2 创建学生模型:")
    student_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 定义蒸馏损失函数
    def distillation_loss(y_true, y_pred):
        # 软标签损失
        teacher_predictions = teacher_model(x_train, training=False)
        soft_target_loss = tf.keras.losses.categorical_crossentropy(
            teacher_predictions, y_pred, from_logits=False
        )
        # 硬标签损失
        hard_target_loss = tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        # 组合损失
        return 0.7 * soft_target_loss + 0.3 * hard_target_loss
    
    # 编译学生模型
    student_model.compile(
        optimizer='adam',
        loss=distillation_loss,
        metrics=['accuracy']
    )
    
    # 训练学生模型
    print("\n4.3 训练学生模型:")
    student_model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)
    
    # 评估学生模型
    student_loss, student_accuracy = student_model.evaluate(x_test, y_test, verbose=0)
    print(f"学生模型 - 损失: {student_loss:.4f}, 准确率: {student_accuracy:.4f}")
    
    # 比较模型大小
    teacher_model_path = os.path.join(script_dir, 'teacher_model')
    student_model_path = os.path.join(script_dir, 'student_model')
    
    teacher_model.save(teacher_model_path)
    student_model.save(student_model_path)
    
    teacher_model_size = os.path.getsize(os.path.join(teacher_model_path, 'saved_model.pb'))
    student_model_size = os.path.getsize(os.path.join(student_model_path, 'saved_model.pb'))
    
    print("\n4.4 模型大小比较:")
    print(f"教师模型大小: {teacher_model_size / 1024:.2f} KB")
    print(f"学生模型大小: {student_model_size / 1024:.2f} KB")

# 5. 模型转换为TFLite
def model_to_tflite():
    print("\n5. 模型转换为TFLite:")
    
    # 创建并训练基础模型
    model = create_base_model()
    
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    # 训练模型
    model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.1)
    
    # 保存模型
    model_path = os.path.join(script_dir, 'model_for_tflite')
    model.save(model_path)
    
    # 转换为TFLite模型
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    
    # 保存TFLite模型
    tflite_model_path = os.path.join(script_dir, 'model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    # 评估TFLite模型
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 测试TFLite模型
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    correct = 0
    for i in range(len(x_test)):
        input_data = np.expand_dims(x_test[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output) == np.argmax(y_test[i]):
            correct += 1
    
    tflite_accuracy = correct / len(x_test)
    print(f"TFLite模型 - 准确率: {tflite_accuracy:.4f}")
    
    # 比较模型大小
    original_model_size = os.path.getsize(os.path.join(model_path, 'saved_model.pb'))
    tflite_model_size = os.path.getsize(tflite_model_path)
    
    print("\n5.1 模型大小比较:")
    print(f"原始模型大小: {original_model_size / 1024:.2f} KB")
    print(f"TFLite模型大小: {tflite_model_size / 1024:.2f} KB")

# 6. 模型推理优化
def model_inference_optimization():
    print("\n6. 模型推理优化:")
    
    # 创建并训练基础模型
    model = create_base_model()
    
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    # 训练模型
    model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.1)
    
    # 1. 基准推理时间
    print("\n6.1 基准推理时间:")
    start_time = time.time()
    for i in range(1000):
        model.predict(np.expand_dims(x_test[i], axis=0))
    end_time = time.time()
    print(f"基准推理时间: {end_time - start_time:.4f}秒")
    
    # 2. 使用tf.function优化
    print("\n6.2 使用tf.function优化:")
    @tf.function
    def predict_fn(x):
        return model(x)
    
    # 预热
    for i in range(10):
        predict_fn(np.expand_dims(x_test[i], axis=0))
    
    start_time = time.time()
    for i in range(1000):
        predict_fn(np.expand_dims(x_test[i], axis=0))
    end_time = time.time()
    print(f"tf.function优化后推理时间: {end_time - start_time:.4f}秒")
    
    # 3. 批处理推理
    print("\n6.3 批处理推理:")
    batch_size = 32
    num_batches = 1000 // batch_size
    
    start_time = time.time()
    for i in range(num_batches):
        batch = x_test[i*batch_size:(i+1)*batch_size]
        model.predict(batch)
    end_time = time.time()
    print(f"批处理推理时间: {end_time - start_time:.4f}秒")

if __name__ == "__main__":
    # 执行所有演示
    model_quantization()
    model_pruning()
    model_distillation()
    model_to_tflite()
    model_inference_optimization()
    
    print("\n" + "=" * 50)
    print("演示完成！")