import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("TensorFlow Gradient Tape 演示")
print("=" * 50)

# 1. 基本自动微分
def basic_autodiff():
    print("\n1. 基本自动微分:")
    
    # 创建变量
    x = tf.Variable(3.0)
    
    # 使用GradientTape记录操作
    with tf.GradientTape() as tape:
        y = x ** 2
    
    # 计算梯度
    dy_dx = tape.gradient(y, x)
    print(f"x = {x.numpy()}, y = {y.numpy()}, dy/dx = {dy_dx.numpy()}")

# 2. 多变量自动微分
def multivariable_autodiff():
    print("\n2. 多变量自动微分:")
    
    # 创建变量
    x = tf.Variable(2.0)
    y = tf.Variable(3.0)
    
    # 使用GradientTape记录操作
    with tf.GradientTape() as tape:
        z = x * y + x ** 2
    
    # 计算梯度
    dz_dx, dz_dy = tape.gradient(z, [x, y])
    print(f"x = {x.numpy()}, y = {y.numpy()}, z = {z.numpy()}")
    print(f"dz/dx = {dz_dx.numpy()}, dz/dy = {dz_dy.numpy()}")

# 3. 嵌套GradientTape
def nested_gradient_tape():
    print("\n3. 嵌套GradientTape:")
    
    # 创建变量
    x = tf.Variable(1.0)
    
    # 嵌套GradientTape计算二阶导数
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            y = x ** 3
        # 一阶导数
        dy_dx = inner_tape.gradient(y, x)
    # 二阶导数
    d2y_dx2 = outer_tape.gradient(dy_dx, x)
    
    print(f"x = {x.numpy()}, y = {y.numpy()}")
    print(f"dy/dx = {dy_dx.numpy()}, d²y/dx² = {d2y_dx2.numpy()}")

# 4. 持久化GradientTape
def persistent_gradient_tape():
    print("\n4. 持久化GradientTape:")
    
    # 创建变量
    x = tf.Variable(2.0)
    y = tf.Variable(3.0)
    
    # 创建持久化GradientTape
    with tf.GradientTape(persistent=True) as tape:
        z1 = x * y
        z2 = x ** 2
        z3 = y ** 2
    
    # 多次使用tape计算梯度
    dz1_dx = tape.gradient(z1, x)
    dz1_dy = tape.gradient(z1, y)
    dz2_dx = tape.gradient(z2, x)
    dz3_dy = tape.gradient(z3, y)
    
    # 删除tape以释放资源
    del tape
    
    print(f"x = {x.numpy()}, y = {y.numpy()}")
    print(f"z1 = {z1.numpy()}, dz1/dx = {dz1_dx.numpy()}, dz1/dy = {dz1_dy.numpy()}")
    print(f"z2 = {z2.numpy()}, dz2/dx = {dz2_dx.numpy()}")
    print(f"z3 = {z3.numpy()}, dz3/dy = {dz3_dy.numpy()}")

# 5. 自定义训练循环
def custom_training_loop():
    print("\n5. 自定义训练循环:")
    
    # 创建线性模型
    class LinearModel(tf.Module):
        def __init__(self):
            self.w = tf.Variable(0.0)
            self.b = tf.Variable(0.0)
        
        def __call__(self, x):
            return self.w * x + self.b
    
    # 创建模型
    model = LinearModel()
    
    # 生成训练数据
    x_train = tf.constant([1.0, 2.0, 3.0, 4.0])
    y_train = tf.constant([2.0, 4.0, 6.0, 8.0])
    
    # 定义损失函数
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 定义优化器
    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    
    # 训练模型
    epochs = 100
    losses = []
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x_train)
            loss = loss_fn(y_train, y_pred)
        
        # 计算梯度
        gradients = tape.gradient(loss, [model.w, model.b])
        
        # 更新参数
        optimizer.apply_gradients(zip(gradients, [model.w, model.b]))
        
        # 记录损失
        losses.append(loss.numpy())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, w: {model.w.numpy():.4f}, b: {model.b.numpy():.4f}")
    
    # 可视化损失
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'training_loss.png'))
    plt.show()

# 6. 梯度裁剪
def gradient_clipping():
    print("\n6. 梯度裁剪:")
    
    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    
    # 生成训练数据
    x_train = tf.random.normal((100, 1))
    y_train = x_train * 2 + tf.random.normal((100, 1)) * 0.1
    
    # 定义优化器
    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    
    # 训练模型（带梯度裁剪）
    epochs = 50
    losses = []
    gradients_norm = []
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x_train)
            loss = tf.reduce_mean(tf.square(y_train - y_pred))
        
        # 计算梯度
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # 计算梯度范数
        grad_norm = tf.linalg.global_norm(gradients)
        gradients_norm.append(grad_norm.numpy())
        
        # 梯度裁剪
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # 更新参数
        optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
        
        # 记录损失
        losses.append(loss.numpy())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Gradient Norm: {grad_norm.numpy():.4f}")
    
    # 可视化损失和梯度范数
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(range(epochs), losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(range(epochs), gradients_norm)
    ax2.set_title('Gradient Norm')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Norm')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'gradient_clipping.png'))
    plt.show()

# 7. 自定义梯度
def custom_gradient():
    print("\n7. 自定义梯度:")
    
    # 定义带有自定义梯度的函数
    @tf.custom_gradient
    def custom_square(x):
        result = x * x
        
        def grad(dy):
            return 2.0 * x * dy
        
        return result, grad
    
    # 测试自定义梯度
    x = tf.Variable(3.0)
    
    with tf.GradientTape() as tape:
        y = custom_square(x)
    
    dy_dx = tape.gradient(y, x)
    print(f"x = {x.numpy()}, y = {y.numpy()}, dy/dx = {dy_dx.numpy()}")

# 8. 多输出梯度
def multiple_outputs():
    print("\n8. 多输出梯度:")
    
    # 创建变量
    x = tf.Variable(2.0)
    
    # 定义多输出函数
    def multi_output_fn(x):
        return x ** 2, x ** 3
    
    # 使用GradientTape
    with tf.GradientTape() as tape:
        y1, y2 = multi_output_fn(x)
    
    # 计算对第一个输出的梯度
    dy1_dx = tape.gradient(y1, x)
    print(f"x = {x.numpy()}, y1 = {y1.numpy()}, dy1/dx = {dy1_dx.numpy()}")
    
    # 重新计算对第二个输出的梯度
    with tf.GradientTape() as tape:
        y1, y2 = multi_output_fn(x)
    
    dy2_dx = tape.gradient(y2, x)
    print(f"x = {x.numpy()}, y2 = {y2.numpy()}, dy2/dx = {dy2_dx.numpy()}")

# 9. 控制流中的梯度
def gradient_in_control_flow():
    print("\n9. 控制流中的梯度:")
    
    # 定义带有控制流的函数
    def conditional_fn(x):
        if x > 0:
            return x ** 2
        else:
            return x * 3
    
    # 测试正数输入
    x_positive = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        y_positive = conditional_fn(x_positive)
    dy_dx_positive = tape.gradient(y_positive, x_positive)
    print(f"x = {x_positive.numpy()}, y = {y_positive.numpy()}, dy/dx = {dy_dx_positive.numpy()}")
    
    # 测试负数输入
    x_negative = tf.Variable(-2.0)
    with tf.GradientTape() as tape:
        y_negative = conditional_fn(x_negative)
    dy_dx_negative = tape.gradient(y_negative, x_negative)
    print(f"x = {x_negative.numpy()}, y = {y_negative.numpy()}, dy/dx = {dy_dx_negative.numpy()}")

# 10. 与tf.function集成
def gradient_with_tf_function():
    print("\n10. 与tf.function集成:")
    
    # 定义带有tf.function的函数
    @tf.function
    def train_step(model, optimizer, loss_fn, x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_fn(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # 创建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    
    # 生成训练数据
    x_train = tf.random.normal((100, 1))
    y_train = x_train * 2 + tf.random.normal((100, 1)) * 0.1
    
    # 定义优化器和损失函数
    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    loss_fn = tf.losses.MeanSquaredError()
    
    # 训练模型
    epochs = 50
    losses = []
    
    for epoch in range(epochs):
        loss = train_step(model, optimizer, loss_fn, x_train, y_train)
        losses.append(loss.numpy())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
    
    # 可视化损失
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), losses)
    plt.title('Training Loss with tf.function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'loss_with_tf_function.png'))
    plt.show()

if __name__ == "__main__":
    # 执行所有演示
    basic_autodiff()
    multivariable_autodiff()
    nested_gradient_tape()
    persistent_gradient_tape()
    custom_training_loop()
    gradient_clipping()
    custom_gradient()
    multiple_outputs()
    gradient_in_control_flow()
    gradient_with_tf_function()
    
    print("\n" + "=" * 50)
    print("演示完成！")