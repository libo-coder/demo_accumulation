# -*- coding: utf-8 -*-
"""
一些常用的 Python 代码合计整理
@author: libo
"""
import cv2
from matplotlib.pyplot import plot

####################################### python 从 txt 中读取坐标并把点画在图片上 #######################################
txt_file = open('/mnt/libo/myself/txt2xml/test.txt')
img = cv2.imread('/mnt/libo/myself/txt2xml/test.png')
sourcelnLine = txt_file.readlines()
points = []
for line in sourcelnLine:
    temp = line.strip('\n').split(' ')
    points.append(temp)
# 每行都是横纵坐标
for i in range(len(points)):
    x = float(points[i][0])
    y = float(points[i][1])
    plot(x, y, 'r')
#####################################################################################################################


################################### python 用 opencv 在图像中画矩形框，加 text，并保存 ##################################
fname = 'path/xxx.jpg'
img = cv2.imread(fname)
# cv2.rectangle(img, 左上，右下，color(B,G,R)，宽度)
cv2.rectangle(img, (10, 50), (50, 100), (0, 255, 0), 4)
font = cv2.FONT_HERSHEY_SIMPLEX
text = 'example'    # 矩形框的说明
# putText(img，text，开始坐标，字体，字体大小，color，粗细)
cv2.putText(img, text, (50, 50), font, 1, (0, 0, 255), 1)
cv2.imwrite('path/newname.jpg', img)
#####################################################################################################################


############################################## PIL与OpenCV转换 ######################################################
import cv2
import numpy as np
from PIL import Image

# PIL 转 OpenCV
cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

# Opencv 转 PIL
pil_img = Image.fromarray(cv2_img).convert('L')
#####################################################################################################################