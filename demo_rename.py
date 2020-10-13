# -*- coding: utf-8 -*-
"""
批量图片重命名
@author: libo
"""
import os
import shutil

path = '/home/libo/myself/rename/1'
filelist = os.listdir(path)                 # 该文件夹下所有的文件（包括文件夹）
count = 1
for file in filelist:   # 遍历所有文件
    Olddir = os.path.join(path, file)       # 原来的文件路径
    if os.path.isdir(Olddir):               # 如果是文件夹则跳过
        continue
    bfn, ext = os.path.splitext(file)
    if ext.lower() not in ['.jpg', '.png']:
        continue
    Newdir = os.path.join(path, 'libo_' + str(count).zfill(3) + '.jpg')
    os.rename(Olddir, Newdir)  # 重命名
    count += 1