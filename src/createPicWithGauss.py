#coding:utf-8
from PIL import Image
import glob,os
import numpy as np

limit = 8

for infile in glob.glob(r'..\pic_raw\*.png'):
    im = Image.open(infile)
    for i in range(im.size[0]):
        gauss = np.random.normal(0, limit, im.size[1])
        for j in range(im.size[1]):
            im.putpixel((i,j), int(im.getpixel((i,j))+ gauss[j]))
    im.save('..\pic_gauss\\'+os.path.splitext(infile)[0][11:]+'.png','PNG')
    #print('..\pic_gauss\\'+os.path.splitext(infile)[0][11:]+'.png','PNG')

    print(infile)