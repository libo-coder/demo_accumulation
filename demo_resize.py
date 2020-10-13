# -*- coding: utf-8 -*-
"""
图片批量 resize
@author: libo
"""
import os.path
import glob
import cv2

def convertjpg(jpgfile, outdir, width=128, height=32):
    src = cv2.imread(jpgfile, cv2.IMREAD_ANYCOLOR)
    try:
        dst = cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(outdir, os.path.basename(jpgfile)), dst)
    except Exception as e:
        print(e)


for jpgfile in glob.glob(r'image/*.jpg'):
    convertjpg(jpgfile, r'image_re')