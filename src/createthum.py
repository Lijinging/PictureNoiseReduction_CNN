# coding:utf-8
from PIL import Image
import glob,os
size=144,144
cnt = 0

for infile in glob.glob(r"../pic_raw/*.bmp"):
    file,ext=os.path.splitext(infile)
    im=Image.open(infile)
    minsize = min(im.size)
    box=(im.size[0]/2-minsize/2, im.size[1]/2-minsize/2,im.size[0]/2+minsize/2, im.size[1]/2+minsize/2)
    im=im.crop(box)
    im.thumbnail(size,Image.ANTIALIAS)
    im=im.convert('L')
    im.save('../pic_gauss/lena.png','PNG')
    cnt=cnt+1
    print(cnt)

for infile in glob.glob(r"../pic_raw/*.png"):
    file,ext=os.path.splitext(infile)
    im=Image.open(infile)
    minsize = min(im.size)
    box=(im.size[0]/2-minsize/2, im.size[1]/2-minsize/2,im.size[0]/2+minsize/2, im.size[1]/2+minsize/2)
    im=im.crop(box)
    im.thumbnail(size,Image.ANTIALIAS)
    im=im.convert('L')
    im.save('../thum/'+str(cnt)+'.png','PNG')
    cnt=cnt+1
    print(cnt)