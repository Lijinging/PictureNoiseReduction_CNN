# coding:utf-8
import tensorflow as tf
from src import getSNR
import scipy.misc
import scipy.io
import os
import numpy as np
from PIL import Image



def testImg(im, pic_save_path = r'../show/after/', img_name = r'lena_big.png', noise_path = r'../show/noise/'):
    pixcnt = im.size[0] * im.size[1]


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


    x = tf.placeholder(tf.float32, shape=[None, im.size[0] * im.size[1]])
    y_ = tf.placeholder(tf.float32, shape=[None, im.size[0] * im.size[1]])

    x_image = tf.reshape(x, [-1, im.size[0], im.size[1], 1])

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
    y = tf.reshape(h_conv4, [1, -1])

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


    def toImageFromRGBArray(arr_r, arr_g, arr_b):
        arr_r = arr_r.astype(int)
        arr_g = arr_g.astype(int)
        arr_b = arr_b.astype(int)
        for [i, j] in [(i, j) for i in range(arr_r.shape[0]) for j in range(arr_r.shape[1])]:
            if arr_r[i][j] < 0:
                arr_r[i][j] = 0
            elif arr_r[i][j] > 255:
                arr_r[i][j] = 255
            if arr_g[i][j] < 0:
                arr_g[i][j] = 0
            elif arr_g[i][j] > 255:
                arr_g[i][j] = 255
            if arr_b[i][j] < 0:
                arr_b[i][j] = 0
            elif arr_b[i][j] > 255:
                arr_b[i][j] = 255

        arr_r = Image.fromarray(arr_r).convert('L')
        arr_g = Image.fromarray(arr_g).convert('L')
        arr_b = Image.fromarray(arr_b).convert('L')

        return Image.merge("RGB", (arr_r, arr_g, arr_b))


    def testImg_sl(in_image, filepath, noisepath):
        p_r, p_g, p_b = in_image.split()

        np_r = np.array(p_r.convert('L')).reshape(1, pixcnt).astype('float32') / 255.0
        np_g = np.array(p_g.convert('L')).reshape(1, pixcnt).astype('float32') / 255.0
        np_b = np.array(p_b.convert('L')).reshape(1, pixcnt).astype('float32') / 255.0

        '''开始预测'''

        r_noise = sess.run(y, feed_dict={x: np_r, keep_prob: 1.0})
        np_r = ((np_r - r_noise) * 255.0).astype(int).reshape(im.size[0], im.size[1])
        r_noise = (r_noise * 255.0 + 128).astype(int).reshape(im.size[0], im.size[1])

        g_noise = sess.run(y, feed_dict={x: np_g, keep_prob: 1.0})
        np_g = ((np_g - g_noise) * 255.0).astype(int).reshape(im.size[0], im.size[1])
        g_noise = (g_noise * 255.0 + 128).astype(int).reshape(im.size[0], im.size[1])

        b_noise = sess.run(y, feed_dict={x: np_b, keep_prob: 1.0})
        np_b = ((np_b - b_noise) * 255.0).astype(int).reshape(im.size[0], im.size[1])
        b_noise = (b_noise * 255.0 + 128).astype(int).reshape(im.size[0], im.size[1])

        im_out = toImageFromRGBArray(np_r, np_g, np_b)
        im_out = im_out.convert('L')
        noise_out = toImageFromRGBArray(abs(r_noise), abs(g_noise), abs(b_noise))
        # im_out.show()
        im_out.save(filepath + img_name, img_name[-3:])
        noise_out.save(noisepath + img_name, img_name[-3:])


    testImg_sl(im, pic_save_path, noise_path)

    sess.close()


