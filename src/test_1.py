# coding:utf-8
import tensorflow as tf
from src import getSNR
import scipy.misc
import scipy.io
import numpy as np
from PIL import Image


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

x = tf.placeholder(tf.float32, shape=[None, 65536])
y_ = tf.placeholder(tf.float32, shape=[None, 65536])

x_image = tf.reshape(x, [-1, 256, 256, 1])

W_conv1 = weight_variable([3, 3, 1, 24])  # 第一层卷积层
b_conv1 = bias_variable([24])  # 第一层卷积层的偏置量
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([3, 3, 24, 24])  # 第二次卷积层
b_conv2 = bias_variable([24])  # 第二层卷积层的偏置量
h_conv2 = tf.nn.relu(batchnormalize(conv2d(h_conv1, W_conv2) + b_conv2))

W_conv3 = weight_variable([3, 3, 24, 24])  # 第三次卷积层
b_conv3 = bias_variable([24])  # 第二层卷积层的偏置量
h_conv3 = tf.nn.relu(batchnormalize(conv2d(h_conv2, W_conv3) + b_conv3))

W_conv4 = weight_variable([3, 3, 24, 1])  # 第四次卷积层
b_conv4 = bias_variable([1])  # 第二层卷积层的偏置量
h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4
y = tf.reshape(h_conv4, [-1, 65536])

keep_prob = tf.placeholder("float")

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.InteractiveSession(config=config)

saver = tf.train.Saver()
save_path = r"..\model_new\model_5120.ckpt"
saver.restore(sess, save_path)

'''
设置输入数据
'''
test_data_raw = scipy.io.loadmat("../data/test.mat")
# 数据归一化
test_data = test_data_raw['data'][:10].astype('float32')
# label:
y_ = test_data_raw['label'][:10, -1]


im_test = np.array(Image.open('../show/pic_with_noise/lena_16.png').convert('L')).reshape(1,65536)
scipy.misc.imsave('../vis/0lena_test.jpg', im_test.reshape(256,256))
im_test = im_test.astype('float32')/255.0

im_label = np.array(Image.open('../pic_raw/lena.png').convert('L')).reshape(1,65536)


'''开始预测'''
pred = sess.run(y, feed_dict={x: im_test, keep_prob:1.0})
im_out = im_test - pred
im_out = (im_out*255.0).astype(int)
for i in range(im_out.shape[0]):
    for j in range(im_out.shape[1]):
        if im_out[i][j]<0:
            im_out[i][j] = 0
        elif im_out[i][j]>255:
            im_out[i][j] = 255
pred = (pred*255.0).astype(int)
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i][j]<0:
            pred[i][j] = -pred[i][j]
        elif pred[i][j]>255:
            pred[i][j] = 255

def testImg(infile = '../show/pic_with_noise/lena_7.png', infile_raw = '../pic_raw/lena.png', filename = 'lena'):
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
    scipy.misc.imsave('../vis/0'+filename+'_test.jpg', im_test.reshape(256, 256))
    scipy.misc.imsave('../vis/1'+filename+'.jpg', im_out.reshape(256, 256))
    scipy.misc.imsave('../vis/3'+filename+'_noise.jpg', pred.reshape(256, 256))
    scipy.misc.imsave('../vis/2'+filename+'_label.jpg', im_label.reshape(256, 256))

    print("Before   ", end=':')
    getSNR.getSNR(Image.open(r'../vis/0lena_test.jpg').convert('L'), Image.open(r'../vis/2lena_label.jpg').convert('L'))
    print("After    ", end=':')
    getSNR.getSNR(Image.open(r'../vis/1lena.jpg').convert('L'), Image.open(r'../vis/2lena_label.jpg').convert('L'))


scipy.misc.imsave('../vis/1lena.jpg', im_out.reshape(256,256))
scipy.misc.imsave('../vis/3lena_noise.jpg', pred.reshape(256,256))
scipy.misc.imsave('../vis/2lena_label.jpg', im_label.reshape(256,256))


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
