import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import SGD

# 读入数据/标签 生成x_train y_train
df = pd.read_csv('data/dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_y_color = [['red' if y else 'blue'] for y in y_data]

x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)

# ([网络结构])  #描述各层网络
model = models.Sequential([
    # 拉直层可以变换张量的尺寸，把输入特征拉直为一维数组，是不含计算参数的层
    layers.Flatten(),
    # 全连接层：Dense(神经元个数，activation = "激活函数“，kernel_regularizer = "正则化方式）
    layers.Dense(11),
    layers.Dense(1, activation='relu')
])

# optimizer = 优化器，loss = 损失函数，metrics = ["准确率”]
model.compile(optimizer=SGD(learning_rate=0.005),
              loss='mse',
              metrics=['accuracy'])

# 加载保存的网络模型
checkpoint_save_path = "./checkpoint/dot.ckpt"
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
history = model.fit(x_train, y_train, batch_size=32, epochs=800,
                    callbacks=[cp_callback])

model.save(checkpoint_save_path)
# 输出模型各层的参数状况
model.summary()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['accuracy']
loss = history.history['loss']
# 第一个代表行数，第二个代表列数，第三个代表索引位置
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
