import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#
#
#
#
#   有问题，但是又不知道哪里错了！！！
#   等到我学成归来，再看！！！
#
#
#
#
#
# 读入数据/标签 生成x_train y_train
df = pd.read_csv('data/dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_y_color = [['red' if y else 'blue'] for y in y_data]

# 取第0列给x1，取第1列给x2
x = x_data[:, 0]
y = x_data[:, 1]
plt.scatter(x, y, color=np.squeeze(x_y_color))
plt.show()

x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

weight_init_std = 0.01
input_size = 2
output_size = 1
hidden_size = 11
# 生成神经网络的参数，输入层为2个神经元，隐藏层为11个神经元，1层隐藏层，输出层为1个神经元
w1 = weight_init_std * np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
w2 = weight_init_std * np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

lr = 0.001  # 学习率
epoch = 800  # 循环轮数
batch_size = 50


def relu(x):
    return np.maximum(0, x)


def relu_gradient(signal):
    return -np.minimum(0, signal)


def gradient(W1, b1, W2, b2, x, t):
    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = np.dot(x, W1) + b1
    z1 = relu(a1)
    y = np.dot(z1, W2) + b2

    # backward
    dy = (y - t) / batch_num
    grads['W2'] = np.dot(z1.T, dy)
    grads['b2'] = np.sum(dy, axis=0)

    da1 = np.dot(dy, W2.T)
    dz1 = relu_gradient(a1) * da1
    grads['W1'] = np.dot(x.T, dz1)
    grads['b1'] = np.sum(dz1, axis=0)
    return grads

# x:输入数据, t:监督数据
def loss(W1, b1, W2, b2, x, t):
    y = predict(W1, b1, W2, b2, x)
    return mse_error(y, t)

def predict(W1, b1, W2, b2, x):
    # 第一层结果
    a1 = np.dot(x, W1) + b1
    # 第一层激活函数输出
    z1 = relu(a1)
    # 第二层结果
    y = np.dot(z1, W2) + b2

    return y


def mse_error(y, t):
    loss = np.mean(np.square(t - y))
    return loss


l = -1
# 训练部分
for epoch in range(epoch):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        # 每20个epoch，打印loss信息
        l = loss(w1, b1, w2, b2, x_batch, y_batch)
        grads = gradient(w1, b1, w2, b2, x_batch, y_batch)
        # 实现梯度更新
        # w1 = w1 - lr * w1_grad
        w1 -= (lr * grads['W1'])
        b1 -= (lr * grads['b1'])
        w2 -= (lr * grads['W2'])
        b2 -= (lr * grads['b2'])

    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(l))

# y1 = predict(w1, b1, w2, b2, x_data[:10])
# print('x_data[:10] :', x_data[:10]," result: ",y1)
