import os
import shutil

from_i = "/Users/denis/PycharmProjects/Segmentation/train/input/"
from_out = "/Users/denis/PycharmProjects/Segmentation/train/output/"

to_1 = "/Users/denis/PycharmProjects/Segmentation/train1/img"

files1=os.listdir(from_i)
files2=os.listdir(from_out)
g="1"

for i in range(1,778):
    shutil.move(from_i+"img"+str(i)+".png", to_1 + str(i)+'/images')
    shutil.move(from_out + "img" + str(i) + ".png", to_1 + str(i)+'/masks')