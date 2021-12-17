import tensorflow as tf
#
# 1.Fetch
# 为了取回操作的输出内容, 可以在使用 Session 对象的 run() 调用 执行图时,
# 传入一些 tensor, 这些 tensor 会帮助你取回结果.
# 在之前的例子里, 我们只取回了单个节点 state, 但是你也可以取回多个 tensor:
# 需要获取的多个 tensor 值,在 op 的一次运行中一起获得(而不是逐个去获取 tensor).
#

# 2.Feed
# 上述示例在计算图中引入了 tensor, 以常量或变量的形式存储.
# TensorFlow 还提供了 feed 机制, 该机制可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁,
# 直接插入一个 tensor.
#
# feed 使用一个 tensor 值临时替换一个操作的输出结果.
# 你可以提供 feed 数据作为 run() 调用的参数. feed 只在调用它的方法内有效, 方法结束, feed 就会消失.
# 最常见的用例是将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
#
tf.compat.v1.disable_eager_execution()

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.compat.v1.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)

# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]

input1 = tf.compat.v1.placeholder(tf.float32)
input2 = tf.compat.v1.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.compat.v1.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))