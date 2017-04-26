#coding:utf-8
from PIL import Image
import glob,os
import numpy as np

limit = 12

for infile in glob.glob(r'..\pic_gauss\*.png'):
    im = Image.open(infile)
    im_raw = Image.open('..\pic_raw\\'+os.path.splitext(infile)[0][13:]+'.png')
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            im.putpixel((i,j), int(im.getpixel((i,j))-im_raw.getpixel((i,j))))
    im.save('..\pic_noise\\'+os.path.splitext(infile)[0][13:]+'.png','PNG')
    #print('..\pic_gauss\\'+os.path.splitext(infile)[0][11:]+'.png','PNG')

    print(infile)