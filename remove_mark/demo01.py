# -*- coding: utf-8 -*-
"""
去除红色印章问题
@author: libo
"""
import cv2
import numpy as np
import time
import pysnooper        # 代码调试利器

# @pysnooper.snoop()
def remove_red_v0(image_src, resize_ratio=-1):
    time0 = time.time()
    if resize_ratio == 1:
        image = image_src
    elif resize_ratio == -1:
        image = cv2.resize(image_src, (640, 480), interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(image_src, (int(image_src.shape[1] * resize_ratio), int(image_src.shape[0] * resize_ratio)), interpolation=cv2.INTER_CUBIC)
    print('resize_time:', time.time() - time0)

    time1 = time.time()
    B_channel, G_channel, R_channel = cv2.split(image)  # 注意cv2.split()返回通道顺序
    _, RedThresh = cv2.threshold(R_channel, 200, 255, cv2.THRESH_BINARY)  # 100, 80
    cv2.imwrite('./res_test/RedThresh.jpg', RedThresh)
    _, BlueThresh = cv2.threshold(B_channel, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./res_test/BlueThresh.jpg', BlueThresh)

    # RedThresh = cv2.adaptiveThreshold(R_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)  # 自适应二值化
    # BlueThresh = cv2.adaptiveThreshold(B_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)  # 自适应二值化

    mask = cv2.bitwise_xor(RedThresh, BlueThresh)
    cv2.imwrite("./res_test/mask.jpg", mask)
    # mask_ = cv2.fastNlMeansDenoising(mask, None, 50, 7, 21)
    # cv2.imwrite('./res_test/mask_denoise.jpg', mask_)
    print('detect_time:', time.time() - time1)

    raise

    time2 = time.time()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, Thresh = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)  # 130
    _, Thresh = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)  # 130
    cv2.imwrite('./res_test/gray.jpg', Thresh)
    result = cv2.bitwise_xor(Thresh, mask)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    print('inpaint_time:', time.time() - time2)
    return result

# @pysnooper.snoop()
# 在原图中去水印  弊端: 不会采用的方式，太耗时！！！
def remove_red_v1(image_src, resize_ratio=-1):
    time0 = time.time()
    if resize_ratio == 1:
        image = image_src
    elif resize_ratio == -1:
        image = cv2.resize(image_src, (640, 480), interpolation=cv2.INTER_CUBIC)  # 获取图片高宽
    else:
        image = cv2.resize(image_src, (int(image_src.shape[1] * resize_ratio), int(image_src.shape[0] * resize_ratio)), interpolation=cv2.INTER_CUBIC)
    print('resize_time:', time.time() - time0)

    time1 = time.time()
    B_channel, G_channel, R_channel = cv2.split(image)  # 注意cv2.split()返回通道顺序
    _, RedThresh = cv2.threshold(R_channel, 200, 255, cv2.THRESH_BINARY)  # 100
    _, BlueThresh = cv2.threshold(B_channel, 150, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_xor(RedThresh, BlueThresh)
    cv2.imwrite("./res_test/mask.jpg", mask)
    mask = cv2.fastNlMeansDenoising(mask, None, 10, 7, 21)
    cv2.imwrite('./res_test/mask_denoise.jpg', mask)

    kernel = np.ones((3, 3), np.uint8)
    dilate_img = cv2.dilate(mask, kernel, iterations=1)
    cv2.imwrite('./res_test/dilate_img.jpg', dilate_img)
    print('detect_time:', time.time() - time1)

    time2 = time.time()
    dst = cv2.inpaint(image, dilate_img, 3, cv2.INPAINT_TELEA)
    print('inpaint_time:', time.time() - time2)
    return dst

# 基于像素的反色中和（处理质量较高）
def remove_red_v2(image_src, resize_ratio=-1):
    time0 = time.time()
    if resize_ratio == 1:
        image = image_src
    elif resize_ratio == -1:
        image = cv2.resize(image_src, (640, 480), interpolation=cv2.INTER_CUBIC)  # 获取图片高宽
    else:
        image = cv2.resize(image_src, (int(image_src.shape[1] * resize_ratio), int(image_src.shape[0] * resize_ratio)), interpolation=cv2.INTER_CUBIC)
    print('resize_time:', time.time() - time0)

    time1 = time.time()
    B_channel, G_channel, R_channel = cv2.split(image)  # 注意cv2.split()返回通道顺序
    _, RedThresh = cv2.threshold(R_channel, 200, 255, cv2.THRESH_BINARY)  # 100
    _, BlueThresh = cv2.threshold(B_channel, 150, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_xor(RedThresh, BlueThresh)
    cv2.imwrite("./res_test/mask.jpg", mask)
    print('detect_time:', time.time() - time1)

    time2 = time.time()
    save = np.zeros(image_src.shape, np.uint8)  # 创建一张空图像用于保存
    # 这一段遍历耗时过大！！！
    for row in range(image_src.shape[0]):
        for col in range(image_src.shape[1]):
            for channel in range(image_src.shape[2]):
                if mask[row, col] == 0:
                    val = 0
                else:
                    reverse_val = 255 - image_src[row, col, channel]
                    val = 255 - reverse_val * 256 / mask[row, col]
                    if val < 0:
                        val = 0
                save[row, col, channel] = val
    print('save_time:', time.time() - time2)
    return save

if __name__ == '__main__':
    img0 = cv2.imread("./test/001.png")  # 以BGR色彩读取图片
    img1 = remove_red_v2(img0, resize_ratio=1)
    cv2.imwrite("./res_test/output.jpg", img1)

    # while True:
    #     cv2.imshow("output", img1)
    #     if cv2.waitKey() == ord('q'):
    #         break  # 按下‘q’键，退出
    # cv2.destroyAllWindows()