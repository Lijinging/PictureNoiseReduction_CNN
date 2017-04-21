# coding:gbk

from PIL import Image

pic_path=r'..\pic_gauss\0.png'
pic_origin_path=r'..\pic_raw\0.png'

pic = Image.open(pic_path)
pic_origin = Image.open(pic_origin_path)



def getSNR(img, img_origin):
    S = 0
    N = 0
    size = img.size
    print(size)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(3):
                S = S + img.getpixel((i,j))**2
                N = N + (img.getpixel((i,j))-img_origin.getpixel((i,j)))**2
    if N==0:
        print("SNR = INT")
    else:
        print("SNR =", float(S)/N)






getSNR(pic, pic_origin)