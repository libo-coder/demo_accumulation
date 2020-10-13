# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import time
import pysnooper

def find_circles(img):
    take = time.time()
    # cv2.imwrite('circles_test/origin.jpg', img)
    h, w = img.shape[0:2]
    print('h1 = %s, w1 = %s' % (h, w))

    coe = 0.25
    dstHeight, dstWidth = int(h * coe), int(w * coe)
    mat = np.float32([[coe, 0, 0], [0, coe, 0]])
    img_resize = cv2.warpAffine(img, mat, (dstWidth, dstHeight))
    h2, w2 = img_resize.shape[0:2]
    print('h2 = %s, w2 = %s' % (h2, w2))
    # cv2.imwrite('circles_test/img_resize.jpg', img_resize)

    time1 = time.time()
    for row in range(h2):
        for col in range(w2):
            if img_resize[row][col][0] < 150 and img_resize[row][col][1] < 150 and img_resize[row][col][2] > 150:           # 127   127     128
                continue
            else:
                img_resize[row][col][0] = (img_resize[row][col][2]*299 + img_resize[row][col][1]*587 + img_resize[row][col][0]*144 + 500) / 1000
                img_resize[row][col][1] = img_resize[row][col][0]
                img_resize[row][col][2] = img_resize[row][col][0]
    cv2.imwrite('circles_test/img_resize2.jpg', img_resize)
    print('pix_mod_time', time.time() - time1)

    B_channel, G_channel, R_channel = cv2.split(img_resize)  # 注意cv2.split()返回通道顺序
    # cv2.imwrite('circles_test/B_channel.jpg', B_channel)
    # cv2.imwrite('G_channel.jpg', G_channel)
    # cv2.imwrite('circles_test/R_channel.jpg', R_channel)

    _, RedThresh = cv2.threshold(R_channel, 200, 255, cv2.THRESH_BINARY)
    _, BlueThresh = cv2.threshold(B_channel, 200, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_xor(RedThresh, BlueThresh)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算，避免边缘连接断裂问题
    cv2.imwrite('circles_test/mask.jpg', mask)

    binary = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)      # 自适应二值化   25, 10
    cv2.imwrite('circles_test/binary.jpg', binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('len(contours) = ', len(contours))

    # img_copy = img_resize.copy()
    # imgResize_copy = np.zeros([h2, w2, 3], np.uint8)
    blocks_contours = []
    pts = []
    for contour in contours:
        mark_flag = True
        contourArea = cv2.contourArea(contour)
        # print('contourArea = ', contourArea)
        # print('contour = ', contour)
        if contourArea < h2 * w2 / 100 or contourArea > h2 * w2 / 10:
            mark_flag = False
        if not mark_flag:
            continue

        # 如果包围框的中心点落在另一个包围框中
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])  # 获取质心信息
        cy = int(M['m01'] / M['m00'])
        pt_center = (cx, cy)
        if cx * cx + cy * cy <= w2 * w2 * 5 / 12:
            mark_flag = False
        for cnt in blocks_contours:
            if cv2.pointPolygonTest(cnt, pt_center, True) > 0:
                mark_flag = False
        if not mark_flag:
            continue
        blocks_contours.append(contour)

        rect = cv2.minAreaRect(contour)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        # print('rect = ', rect)
        box = cv2.boxPoints(rect)       # 获取最小外接矩形的4个顶点坐标
        box = np.int0(box)
        # print('box = ', box)
        cv2.drawContours(img_resize, [box], 0, (255, 0, 0), 1)
        cv2.imwrite('circles_test/contours_rect.jpg', img_resize)

        x_min = box[1][0]
        y_min = box[1][1]
        x_max = box[3][0]
        y_max = box[3][1]
        pts.append([x_min, y_min, x_max, y_max])
        print('pts = ', pts)

    print('len(blocks_contours) = ', len(blocks_contours))

    if pts:
        tmp = 0
        for pt in pts:
            print('pt = ', pt)
            roi = img[pt[1]*4: pt[3]*4, pt[0]*4: pt[2]*4]
            tmp += 1
            cv2.imwrite('circles_test/roi' + str(tmp) + '.jpg', roi)
            B_roi, G_roi, R_roi = cv2.split(roi)
            _, RedThresh_roi = cv2.threshold(R_roi, 128, 255, cv2.THRESH_BINARY)        # 100
            cv2.imwrite('circles_test/RedThresh_roi' + str(tmp) + '.jpg', RedThresh_roi)
            # 膨胀操作
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))         # (3, 3)
            erode = cv2.erode(RedThresh_roi, element)
            cv2.imwrite('circles_test/erode_roi' + str(tmp) + '.jpg', erode)

            erodeImg = np.expand_dims(erode, axis=2)
            erodeImg = np.concatenate((erodeImg, erodeImg, erodeImg), axis=-1)
            cv2.imwrite('circles_test/erodeImg' + str(tmp) + '.jpg', erodeImg)

            img[pt[1] * 4: pt[3] * 4, pt[0] * 4: pt[2] * 4] = erodeImg
            cv2.imwrite('circles_test/imgOrigin' + str(tmp) + '.jpg', img)


if __name__ == "__main__":
    # img_path = 'table_test/test07.JPG'
    img_path = 'table_test/red01.jpg'
    src = cv2.imread(img_path)
    find_circles(src)