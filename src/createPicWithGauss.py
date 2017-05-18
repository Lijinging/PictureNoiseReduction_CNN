#coding:utf-8
from PIL import Image
import glob,os
import numpy as np
import _thread

limit = 16

for infile in glob.glob(r'..\show\Test12\*.png'):
    im = Image.open(infile).convert('L')
    for i in range(im.size[0]):
        gauss = np.random.normal(0, limit, im.size[1])
        for j in range(im.size[1]):
            im.putpixel((i,j), int(im.getpixel((i,j))+ gauss[j]))
    im.save('..\show\Test12_with_noise\\'+os.path.splitext(infile)[0][15:]+'.png','PNG')
    #print('..\pic_gauss\\'+os.path.splitext(infile)[0][11:]+'.png','PNG')

    print(infile)

#def addGauss(infilelist, length, start, step):
#    for i in range(start, start+step):
#        infile = infilelist[i]
#        if i >= length:
#            return
#        im = Image.open(infile)
#        for i in range(im.size[0]):
#            gauss = np.random.normal(0, limit, im.size[1])
#            for j in range(im.size[1]):
#                im.putpixel((i, j), int(im.getpixel((i, j)) + gauss[j]))
#        im.save('..\pic_gauss\\' + os.path.splitext(infile)[0][11:] + '.png', 'PNG')
#        # print('..\pic_gauss\\'+os.path.splitext(infile)[0][11:]+'.png','PNG')
#        print(infile)
#
#
#infilelist = glob.glob(r'..\pic_raw\*.png')
#length = len(infilelist)
#addGauss(infilelist, length, 0, length)

#try:
#    _thread.start_new_thread(addGauss, (infilelist, length, 0, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 100, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 200, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 300, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 400, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 500, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 600, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 700, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 800, 100))
#    _thread.start_new_thread(addGauss, (infilelist, length, 900, 100))
#except:
#   print ("Error: 无法启动线程")
#
#while(1):
#    pass