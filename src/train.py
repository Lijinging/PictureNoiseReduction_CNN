# coding:utf-8

import tensorflow as tf
import scipy.io as scio
import numpy as np

'''
设置输入数据
'''
train_data_raw = scio.loadmat("../data/train_rand_6_25.mat")
test_data_raw = scio.loadmat("../data/test_rand_6_25.mat")
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
    return X
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


def bias(name, shape, bias_start=0.0, trainable=True):
    return tf.get_variable(name, shape, tf.float32, trainable=trainable,
                           initializer=tf.constant_initializer(bias_start, dtype=tf.float32))


def batch_norm(value, is_train=True, name='batch_norm', eps=1e-5, momentum=0.9):
    with tf.variable_scope(name):
        ema = tf.train.ExponentialMovingAverage(decay=momentum)
        shape = value.get_shape().as_list()[-1]
        beta = bias('beta', [shape], bias_start=0.0)
        gamma = bias('gamma', [shape], bias_start=1.0)
        if is_train:
            batch_mean, batch_variance = tf.nn.moments(value, list(range(len(value.get_shape().as_list()) - 1)),
                                                       name='moments')
            moving_mean = bias('moving_mean', [shape], 0.0, False)
            moving_variance = bias('moving_variance', [shape], 1.0, False)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                ema_apply_op = ema.apply([batch_mean, batch_variance])
            assign_mean = moving_mean.assign(ema.average(batch_mean))
            assign_variance = moving_variance.assign(ema.average(batch_variance))
            with tf.control_dependencies([ema_apply_op]):
                mean, variance = tf.identity(batch_mean), tf.identity(batch_variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_normalization(value, mean, variance, beta, gamma, 1e-5)
        else:
            mean = bias('moving_mean', [shape], 0.0, False)
            variance = bias('moving_variance', [shape], 1.0, False)
            return tf.nn.batch_normalization(value, mean, variance, beta, gamma, eps)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.InteractiveSession(config=config)

x = tf.placeholder(tf.float32, shape=[None, 65536])
y_ = tf.placeholder(tf.float32, shape=[None, 65536])

x_image = tf.reshape(x, [-1, 256, 256, 1])

W_conv1 = weight_variable([5, 5, 1, 24])  # 第一层卷积层
b_conv1 = bias_variable([24])  # 第一层卷积层的偏置量
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([5, 5, 24, 24])  # 第二次卷积层
b_conv2 = bias_variable([24])  # 第二层卷积层的偏置量
h_conv2 = tf.nn.relu(batchnormalize(conv2d(h_conv1, W_conv2) + b_conv2))

W_conv3 = weight_variable([5, 5, 24, 24])  # 第三次卷积层
b_conv3 = bias_variable([24])  # 第二层卷积层的偏置量
h_conv3 = tf.nn.relu(batchnormalize(conv2d(h_conv2, W_conv3) + b_conv3))

W_conv4 = weight_variable([5, 5, 24, 1])  # 第四次卷积层
b_conv4 = bias_variable([1])  # 第二层卷积层的偏置量
h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4
y = tf.reshape(h_conv4, [-1, 65536])

keep_prob = tf.placeholder("float")

cross_entropy = tf.reduce_sum((y_ - y) ** 2)
train_step = tf.train.AdamOptimizer(learning_rate=2e-4, epsilon=1e-8).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(cross_entropy, "float"))
saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
saver.restore(sess, r"..\model_rand\model_540.ckpt")


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
    size = 50
    # batch = mnist.train.next_batch(100)
    batch = next_batch(train_data, train_label, i * size, size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    if (i+1) % 20 == 0:
        print((i + 1), end=":")
        print("test loss %g" % accuracy.eval(feed_dict={
            x: test_data, y_: test_label, keep_prob: 1.0}))
    if i % 20 == 0:
        save_path = r"..\model_rand\model_%d.ckpt" % i
        saver.save(sess, save_path)

save_path = r"..\model_rand\model.ckpt"
saver.save(sess, save_path)

sess.close()
