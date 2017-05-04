#coding:utf-8

import os
import scipy.io as sio
import numpy
from PIL import Image

dir_path = r'../pic_gauss/'
raw_path = r'../pic_raw/'
mat_path = r'../data/'
testNum = 100


# 以下创建mat文件
train_label = numpy.zeros((0, 65536), dtype="int")
train_data = numpy.zeros((0, 65536), dtype="int")
test_label = numpy.zeros((0, 65536), dtype="int")
test_data = numpy.zeros((0, 65536), dtype="int")

filelist = os.listdir(raw_path)
print(len(filelist))
cnt = 0
limit = 16

for infile in filelist:

    cnt = cnt + 1
    print(cnt, infile)

    gauss = numpy.random.normal(0, limit, 65536).astype(int)
    if cnt >= testNum:
        img_train_data = numpy.array(Image.open(raw_path+infile)).reshape(1, 65536)+gauss
        train_data = numpy.row_stack((train_data, img_train_data))
        img_train_label = gauss
        train_label = numpy.row_stack((train_label, img_train_label))
    else:
        img_test_data = numpy.array(Image.open(raw_path + infile)).reshape(1, 65536)+gauss
        test_data = numpy.row_stack((test_data, img_test_data))
        img_test_label = gauss
        test_label = numpy.row_stack((test_label, img_test_label))




sio.savemat(mat_path+'train.mat', {'data': train_data, 'label': train_label})
sio.savemat(mat_path+'test.mat', {'data': test_data, 'label': test_label})