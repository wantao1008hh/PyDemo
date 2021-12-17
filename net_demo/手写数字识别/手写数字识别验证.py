import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras import layers, models, losses
import cv2.cv2 as cv2

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


# x_test_save_path = "/Users/yangminghui/Documents/训练数据集/MNIST_data/x_test.npy"
# x_test = np.load(x_test_save_path)
# np.random.shuffle(x_test)
# x_test = tf.cast(x_test, tf.float32)
# img_arr=x_test[0]
#自己随便写一张图片,测试集非常nice，但是真正自己手写的就识别率很低
png = "/Users/yangminghui/Documents/训练数据集/MNIST_data/1.png"
img = Image.open(png)
img_resize = img.resize((28, 28), Image.ANTIALIAS)
img_arr = np.array(img_resize.convert('L'))
print('img_arr: ', img_arr.shape)
plt.imshow(img_arr)
plt.show()
x_predict = img_arr[tf.newaxis, ...]
print('x_predict: ', x_predict.shape)
result = model.predict(x_predict)
pred = tf.argmax(result, axis=1)
print("pred: ", pred)
