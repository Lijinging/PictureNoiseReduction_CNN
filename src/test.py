# coding:utf-8
import tensorflow as tf
import scipy.io
import scipy.misc
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



x = tf.placeholder(tf.float32, shape=[None, 65536])
y_ = tf.placeholder(tf.float32, shape=[None, 65536])

x_image = tf.reshape(x, [-1, 256, 256, 1])

W_conv1 = weight_variable([5, 5, 1, 20])  # 第一层卷积层
b_conv1 = bias_variable([20])  # 第一层卷积层的偏置量
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)  # 第一次池化层

#W_conv2 = weight_variable([5, 5, 20, 50])  # 第二次卷积层
#b_conv2 = bias_variable([50])  # 第二层卷积层的偏置量
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)  # 第二曾池化层

# W_conv2 = weight_variable([5, 5, 20, 1])  # 第二次卷积层
# b_conv2 = bias_variable([1])  # 第二层卷积层的偏置量
# h_conv2 = tf.nn.tanh(conv2d(h_conv1, W_conv2) + b_conv2)
# #h_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
# y = tf.reshape(h_conv2, [-1, 65536])

W_conv2 = weight_variable([5, 5, 20, 20])  # 第二次卷积层
b_conv2 = bias_variable([20])  # 第二层卷积层的偏置量
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#h_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

W_conv3 = weight_variable([5, 5, 20, 1])  # 第二次卷积层
b_conv3 = bias_variable([1])  # 第二层卷积层的偏置量
h_conv3 = tf.nn.tanh(conv2d(h_conv2, W_conv3) + b_conv3)
#h_conv3 = conv2d(h_conv1, W_conv2) + b_conv2
y = tf.reshape(h_conv3, [-1, 65536])

#W_fc1 = weight_variable([2 * 2 * 50, 500])  # 全连接层
#b_fc1 = bias_variable([500])  # 偏置量
#h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 2 * 50])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#

keep_prob = tf.placeholder("float")

#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#W_fc2 = weight_variable([500, 32])
#b_fc2 = bias_variable([32])
#y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

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


im_test = np.array(Image.open('../pic_gauss/lena.png').convert('L')).reshape(1,65536)
scipy.misc.imsave('../vis/lena_test.jpg', im_test.reshape(256,256))
im_test = im_test.astype('float32') / 255.0

im_label = np.array(Image.open('../pic_raw/lena.png').convert('L')).reshape(1,65536)


'''开始预测'''
pred = sess.run(y, feed_dict={x: im_test, keep_prob:1.0})
im_out = im_test - pred
im_out = im_out*255.0
im_out = im_out.astype(int)
for i in range(im_out.shape[0]):
    for j in range(im_out.shape[1]):
        if im_out[i][j]<0:
            im_out[i][j] = 0
        elif im_out[i][j]>255:
            im_out[i][j] = 255
pred = pred*255.0
pred = pred.astype(int)
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i][j]<0:
            pred[i][j] = -pred[i][j]
        elif pred[i][j]>255:
            pred[i][j] = 255


scipy.misc.imsave('../vis/lena.jpg', im_out.reshape(256,256))
scipy.misc.imsave('../vis/lena_noise.jpg', pred.reshape(256,256))
scipy.misc.imsave('../vis/lena_label.jpg', im_label.reshape(256,256))

print(pred)
print(im_out)
print(im_label)
#print(LABELS[np.argmax(pred, 1)])
#print(LABELS[y_])



sess.close()
