#coding:utf-8

import os
import scipy.io as sio
import numpy
from PIL import Image

dir_path = r'../pic_gauss/'
mat_path = r'../data/'
mat_name = 'data.mat'
testNum = 100


# 以下创建mat文件
label = numpy.zeros((0, 65536), dtype="int")
data = numpy.zeros((0, 65536), dtype="int")

filelist = os.listdir(dir_path)
print(len(filelist))
cnt = 0

for infile in filelist:
    if cnt >= testNum:
        img_data = numpy.array(Image.open(dir_path+infile)).reshape(1, 65536)
        data = numpy.row_stack((data, img_data))
    else:
        img_label = numpy.array(Image.open(dir_path+infile)).reshape(1, 65536)
        label = numpy.row_stack((data, img_label))
    cnt = cnt+1
    print(cnt)


sio.savemat(mat_path+mat_name, {'data': data, 'label': label})