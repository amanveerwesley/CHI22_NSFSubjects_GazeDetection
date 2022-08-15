from os import listdir
import cv2

p = 'Frames/Frames2000_WithAnnotations/'

lf = os.listdir(p)


for n in range(0, len(lf)):
    print(n)
    img = cv2.imread(p + lf[n])
    img = cv2.rectangle(img, (0, 445), (125, 480), (0,0,0), -1)
    cv2.imwrite('Frames/Frames2000_NoAnnotations/' + lf[n], img)