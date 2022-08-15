from asyncore import read
from ntpath import join
import os
import cv2

lf = os.listfiles()

p = 'Frames/Test/'

for n in range(0, len(lf)):
    print(n)
    img = cv2.imread(p + lf[n])
    