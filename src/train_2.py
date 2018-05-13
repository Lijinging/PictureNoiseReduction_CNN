# coding:utf-8

import tensorflow as tf
import scipy.io as scio
import numpy as np

picSize = 144, 144
pixelNum = picSize[0] * picSize[1]

'''
设置输入数据
'''
train_data_raw = scio.loadmat("../data/train.mat")
test_data_raw = scio.loadmat("../data/test.mat")
# 数据归一化
train_data = train_data_raw['data'].astype('float32') / 255.0
test_data = test_data_raw['data'].astype('float32') / 255.0

train_label = train_data_raw['label'].astype('float32') / 255.0
test_label = test_data_raw['label'].astype('float32') / 255.0

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0, 1, 2])
        std = tf.reduce_mean(tf.square(X - mean), [0, 1, 2])
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, -1])
            X = X * g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X - mean), 0)
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, -1])
            b = tf.reshape(b, [1, -1])
            X = X * g + b

    else:
        raise NotImplementedError

    return X

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.InteractiveSession(config=config)

x = tf.placeholder(tf.float32, shape=[None, pixelNum])
y_ = tf.placeholder(tf.float32, shape=[None, pixelNum])

x_image = tf.reshape(x, [-1, picSize[0], picSize[1], 1])

W_conv1 = weight_variable([7, 7, 1, 30])  # 第一层卷积层
b_conv1 = bias_variable([30])  # 第一层卷积层的偏置量
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([7, 7, 30, 30])  # 第二次卷积层
b_conv2 = bias_variable([30])  # 第二层卷积层的偏置量
h_conv2 = tf.nn.relu(batchnormalize(conv2d(h_conv1, W_conv2) + b_conv2))

W_conv3 = weight_variable([5, 5, 30, 30])  # 第三次卷积层
b_conv3 = bias_variable([24])  # 第二层卷积层的偏置量
h_conv3 = tf.nn.relu(batchnormalize(conv2d(h_conv2, W_conv3) + b_conv3))

W_conv4 = weight_variable([5, 5, 30, 1])  # 第四次卷积层
b_conv4 = bias_variable([1])  # 第二层卷积层的偏置量
h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4
y = tf.reshape(h_conv4, [-1, pixelNum])

keep_prob = tf.placeholder("float")

cross_entropy = tf.reduce_sum((y_ - y)**2)
train_step = tf.train.AdamOptimizer(2e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(cross_entropy, "float"))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
#saver.restore(sess, r"..\model_2\model.ckpt")


def next_batch(data, label, begin, length):
    if begin >= data.shape[0]:
        begin = begin % data.shape[0]
    if begin + length > data.shape[0]:
        add = next_batch(data, label, 0, length - (data.shape[0] - begin))
        add[0] = np.row_stack((data[begin:], add[0]))
        add[1] = np.row_stack((label[begin:], add[1]))
    else:
        add = [data[begin:begin + length], label[begin:begin + length]]
    return add


for i in range(9000):
    size = 10
    # batch = mnist.train.next_batch(100)
    batch = next_batch(train_data, train_label, i * size, size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    if i % 50 == 0:
        print(i, end=":")
        print("test loss %g" % accuracy.eval(feed_dict={
            x: test_data, y_: test_label, keep_prob: 1.0}))
    if i % 20 == 0:

        save_path = r"..\model_2\model_%d.ckpt" % i
        saver.save(sess, save_path)



save_path = r"..\model_2\model.ckpt"
saver.save(sess, save_path)


sess.close()
