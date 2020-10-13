# coding=utf-8
"""
创建 LMDB 数据
@author: libo
"""
import fire
import os
import lmdb
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:        # 数组的具体的值映射到具体的 0-255 的 uint8 类型
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)      # 通过这个二进制编码获得图像一维的ndarray数组信息
    # print('----------------------------:', imageBuf.shape)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)      # 通过数组进行图像获取二维32*280大小
    # print('============================:', img.shape)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

"""
def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k in cache:
            # print('1111111111111111111:', k)

            v = cache[k]            # value存放的是图片和标签的二进制编码信息
            # print('2222222222222222222:', v)

            txn.put(k.encode(), v)  # key转换为二进制编码信息，label-000000001  ->  b'label-000000001'
            # print('3333333333333333333:', k.encode())

            ####最终存放的结果:  b'image-*'  :  b'图片的具体的二进制编码信息'
            #:  b'label-*'  :  b'标签文字的具体二进制编码信息'
"""

def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)      # 如果 train 文件夹下没有 data.mbd 或 lock.mdb 文件，则会生成一个空的
    env = lmdb.open(outputPath, map_size=1099511627776)     # map_size: 定义的是一个 lmdb 文件的存储容量，（定义1T）
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()                     # 以二进制文件打开进行读取
        # print('---------------------:', imageBin)     # 获取的是具体图片的二进制编码信息

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split(' ')       # datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        ''' only use alphanumeric data '''
        # if re.search('[^a-zA-Z0-9]', label):
        #     print(label)
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt      # image-000000001(when cnt = 1)
        labelKey = 'label-%09d'.encode() % cnt      # key从1开始，这与后续lmdb信息获取阶段的代码相对应
        cache[imageKey] = imageBin                  # 存放的是图片具体的二进制编码信息
        cache[labelKey] = label.encode()            # 存放label的二进制编码信息

        if cnt % 1000 == 0:
            writeCache(env, cache)                  # cache字典每次存放1000个字符
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))  # 每1000次进行一次打印

        cnt += 1

    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)
