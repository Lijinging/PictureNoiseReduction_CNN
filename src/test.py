# coding:utf-8
import tensorflow as tf
from src import getSNR
import scipy.misc
import scipy.io
import numpy as np
from PIL import Image

picSize = 144, 144
pixelNum = picSize[0] * picSize[1]

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




x = tf.placeholder(tf.float32, shape=[None, pixelNum])
y_ = tf.placeholder(tf.float32, shape=[None, pixelNum])

x_image = tf.reshape(x, [-1, picSize[0], picSize[1], 1])

W_conv1 = weight_variable([5, 5, 1, 30])  # 第一层卷积层
b_conv1 = bias_variable([30])  # 第一层卷积层的偏置量
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([5, 5, 30, 30])  # 第二次卷积层
b_conv2 = bias_variable([30])  # 第二层卷积层的偏置量
# h_conv2 = tf.nn.relu(batchnormalize(conv2d(h_conv1, W_conv2) + b_conv2))
h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, W_conv2) + b_conv2))

W_conv3 = weight_variable([5, 5, 30, 30])  # 第三次卷积层
b_conv3 = bias_variable([30])  # 第二层卷积层的偏置量
# h_conv3 = tf.nn.relu(batchnormalize(conv2d(h_conv2, W_conv3) + b_conv3))
h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, W_conv3) + b_conv3))

W_conv4 = weight_variable([5, 5, 30, 1])  # 第四次卷积层
b_conv4 = bias_variable([1])  # 第二层卷积层的偏置量
h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4
y = tf.reshape(h_conv4, [-1, pixelNum])

keep_prob = tf.placeholder("float")

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.InteractiveSession(config=config)

saver = tf.train.Saver()
save_path = r"..\model\model.ckpt"
saver.restore(sess, save_path)

'''
设置输入数据
'''
test_data_raw = scipy.io.loadmat("../data/test.mat")
# 数据归一化
test_data = test_data_raw['data'][:10].astype('float32') / 255.0
# label:
y_ = test_data_raw['label'][:10, -1]

im_test = np.array(Image.open('../pic_gauss/lena.png').convert('L')).reshape(1, pixelNum)
scipy.misc.imsave('../vis/0lena_test.jpg', im_test.reshape(picSize[0], picSize[1]))
im_test = im_test.astype('float32') / 255.0

im_label = np.array(Image.open('../pic_raw/lena.png').convert('L')).reshape(1, pixelNum)

'''开始预测'''
pred = sess.run(y, feed_dict={x: im_test, keep_prob: 1.0})
im_out = im_test - pred
im_out = im_out * 255.0
im_out = im_out.astype(int)
for i in range(im_out.shape[0]):
    for j in range(im_out.shape[1]):
        if im_out[i][j] < 0:
            im_out[i][j] = 0
        elif im_out[i][j] > 255:
            im_out[i][j] = 255
pred = pred * 255.0
pred = pred.astype(int)
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i][j] < 0:
            pred[i][j] = -pred[i][j]
        elif pred[i][j] > 255:
            pred[i][j] = 255


def testImg(infile='../pic_gauss/lena.png', infile_raw='../pic_raw/lena.png', filename='lena'):
    im_test = np.array(Image.open(infile).convert('L')).reshape(1, 65536)
    scipy.misc.imsave(infile, im_test.reshape(256, 256))
    im_test = im_test.astype('float32') / 255.0

    im_label = np.array(Image.open(infile_raw).convert('L')).reshape(1, 65536)

    '''开始预测'''
    pred = sess.run(y, feed_dict={x: im_test, keep_prob: 1.0})
    im_out = im_test - pred
    im_out = im_out * 255.0
    im_out = im_out.astype(int)
    for i in range(im_out.shape[0]):
        for j in range(im_out.shape[1]):
            if im_out[i][j] < 0:
                im_out[i][j] = 0
            elif im_out[i][j] > 255:
                im_out[i][j] = 255
    pred = pred * 255.0
    pred = pred.astype(int)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i][j] < 0:
                pred[i][j] = -pred[i][j]
            elif pred[i][j] > 255:
                pred[i][j] = 255
    scipy.misc.imsave('../vis/0' + filename + '_test.jpg', im_test.reshape(256, 256))
    scipy.misc.imsave('../vis/1' + filename + '.jpg', im_out.reshape(256, 256))
    scipy.misc.imsave('../vis/3' + filename + '_noise.jpg', pred.reshape(256, 256))
    scipy.misc.imsave('../vis/2' + filename + '_label.jpg', im_label.reshape(256, 256))

    print("Before   ", end=':')
    getSNR.getSNR(Image.open(r'../vis/0lena_test.jpg').convert('L'), Image.open(r'../vis/2lena_label.jpg').convert('L'))
    print("After    ", end=':')
    getSNR.getSNR(Image.open(r'../vis/1lena.jpg').convert('L'), Image.open(r'../vis/2lena_label.jpg').convert('L'))


scipy.misc.imsave('../vis/1lena.jpg', im_out.reshape(256, 256))
scipy.misc.imsave('../vis/3lena_noise.jpg', pred.reshape(256, 256))
scipy.misc.imsave('../vis/2lena_label.jpg', im_label.reshape(256, 256))

print(pred)
print(im_out)
print(im_label)
print("Before   ", end=':')
getSNR.getSNR(Image.open(r'../vis/0lena_test.jpg').convert('L'), Image.open(r'../vis/2lena_label.jpg').convert('L'))
print("After    ", end=':')
getSNR.getSNR(Image.open(r'../vis/1lena.jpg').convert('L'), Image.open(r'../vis/2lena_label.jpg').convert('L'))
print("Ps       ", end=':')
getSNR.getSNR(Image.open(r'../vis/lena_ps.jpg').convert('L'), Image.open(r'../vis/2lena_label.jpg').convert('L'))

print("----------------------------------")

testImg(infile=r'../pic_gauss/279.png', infile_raw=r'../pic_raw/279.png', filename='new')

sess.close()
