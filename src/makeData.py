#coding:utf-8

import os
import scipy.io as sio
import numpy
from PIL import Image
import random

dir_path = r'../pic_gauss/'
raw_path = r'../pic_raw/'
mat_path = r'../data/'
picSize = 144, 144
pixelNum = picSize[0] * picSize[1]
testNum = 100


# 以下创建mat文件
train_label = numpy.zeros((0, pixelNum), dtype="int")
train_data = numpy.zeros((0, pixelNum), dtype="int")
test_label = numpy.zeros((0, pixelNum), dtype="int")
test_data = numpy.zeros((0, pixelNum), dtype="int")

filelist = os.listdir(raw_path)
filelist = numpy.array(filelist)
numpy.random.shuffle(filelist)
print(len(filelist))
cnt = 0
limit_u = 16
limit_d = 16

for infile in filelist:

    cnt = cnt + 1
    limit = random.randint(limit_d, limit_u)
    print(cnt, infile, limit)


    gauss = numpy.random.normal(0, limit, pixelNum).astype(int)
    if cnt >= testNum:
        img_train_data = numpy.array(Image.open(raw_path+infile)).reshape(1, pixelNum)+gauss
        train_data = numpy.row_stack((train_data, img_train_data))
        img_train_label = gauss
        train_label = numpy.row_stack((train_label, img_train_label))
    else:
        img_test_data = numpy.array(Image.open(raw_path + infile)).reshape(1, pixelNum)+gauss
        test_data = numpy.row_stack((test_data, img_test_data))
        img_test_label = gauss
        test_label = numpy.row_stack((test_label, img_test_label))


sio.savemat(mat_path+'train_rand_' + str(limit_u) + '_' + str(limit_d) + '.mat', {'data': train_data, 'label': train_label})
sio.savemat(mat_path+'test_rand_' + str(limit_u) + '_' + str(limit_d) + '.mat', {'data': test_data, 'label': test_label})
