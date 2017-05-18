from src import test_rgb_big
import os
from PIL import Image
from src import getSNR

pic_with_noise_path = r'../show/Test12_with_noise/'
pic_save_path = r'../show/Test12_after/'
noise_path = r'../show/noise/'
pic_raw_path = r'../show/Test12/'
infile = '09.png'
print(infile)
#test_rgb_big.testImg(Image.open(pic_with_noise_path + infile).convert('RGB'), pic_save_path, infile, noise_path)
print(getSNR.getSNR(Image.open(pic_save_path + infile), Image.open(pic_raw_path + infile)))

for infile in os.listdir(pic_save_path):
    print(infile)
    print(getSNR.getSNR(Image.open(pic_save_path + infile), Image.open(pic_raw_path + infile)))
