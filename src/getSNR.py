# coding:gbk

from PIL import Image
import numpy

pic_path=r'..\pic_gauss\28.png'
pic_origin_path=r'..\pic_raw\28.png'

pic = Image.open(pic_path).convert("L")
pic_origin = Image.open(pic_origin_path).convert("L")



def getSNR(img, img_origin):
    S = 0
    N = 0
    size = img.size
    print(size)
    for i in range(size[0]):
        for j in range(size[1]):
            S = S + img.getpixel((i,j))**2
            N = N + (img.getpixel((i,j))-img_origin.getpixel((i,j)))**2
    if N==0:
        print("SNR = INF")
    else:
        print("SNR =", 10*numpy.log10(float(S)/N))






getSNR(pic, pic_origin)