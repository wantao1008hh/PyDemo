import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras import layers, models, losses, callbacks

x_train_save_path = "/Users/yangminghui/Documents/训练数据集/MNIST_data/x_train.npy"
y_train_save_path = "/Users/yangminghui/Documents/训练数据集/MNIST_data/y_train.npy"

x_test_save_path = "/Users/yangminghui/Documents/训练数据集/MNIST_data/x_test.npy"
y_test_save_path = "/Users/yangminghui/Documents/训练数据集/MNIST_data/y_test.npy"

x_train = np.load(x_train_save_path)
y_train = np.load(y_train_save_path)
x_test = np.load(x_test_save_path)
y_test = np.load(y_test_save_path)

print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)

np.random.seed(100)
np.random.shuffle(x_train)
np.random.seed(100)
np.random.shuffle(y_train)

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)


# ([网络结构])  #描述各层网络
model = models.Sequential([
    # 拉直层可以变换张量的尺寸，把输入特征拉直为一维数组，是不含计算参数的层
    layers.Flatten(),
    # 全连接层：Dense(神经元个数，activation = "激活函数“，kernel_regularizer = "正则化方式）
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# optimizer = 优化器，loss = 损失函数，metrics = ["准确率”]
model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 加载保存的网络模型
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)


# 回调函数
cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                        save_weights_only=True,
                                        save_best_only=True)
# 训练集的输入特征，训练集的标签，
# batch_size,  #每一个batch的大小
# epochs,   #迭代次数
# validation_data = (测试集的输入特征，测试集的标签），
# validation_split = 从测试集中划分多少比例给训练集，
# validation_freq = 测试的epoch间隔数
history = model.fit(x_train, y_train, batch_size=32, epochs=5,
                    validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

model.save(checkpoint_save_path)
# 输出模型各层的参数状况
model.summary()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
#第一个代表行数，第二个代表列数，第三个代表索引位置
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
