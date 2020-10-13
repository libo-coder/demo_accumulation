# coding=utf-8
"""
OCR 检测后对返回的 box 坐标进行排序，使其坐标顺序为（左上，右上，右下，左下）
@author: libo
"""
import numpy as np
import copy

def CalcEuclideanDistance(point1, point2):
    """ 计算两点之间的距离 """
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)
    return distance

def CalcFourthPoint(point1, point2, point3):
    """ 已知平行四边形三个点，求第四个点 """
    D = (point1[0] + point2[0] - point3[0], point1[1] + point2[1] - point3[1])
    return D

def adjust(polys):
    new_polys = []
    for poly in polys:
        poly = poly.astype(np.int32)
        x1 = poly[0][0]
        y1 = poly[0][1]
        x2 = poly[1][0]
        y2 = poly[1][1]
        x3 = poly[2][0]
        y3 = poly[2][1]
        x4 = poly[3][0]
        y4 = poly[3][1]

        L2_left = CalcEuclideanDistance([x1, y1], [x4, y4])
        L2_right = CalcEuclideanDistance([x2, y2], [x3, y3])
        left_based = True if L2_left > L2_right else False

        if left_based:
            if x2 > x3:
                [x3, y3] = CalcFourthPoint([x4, y4], [x2, y2], [x1, y1])
            else:
                [x2, y2] = CalcFourthPoint([x1, y1], [x3, y3], [x4, y4])
        else:
            if x1 < x4:
                [x4, y4] = CalcFourthPoint([x3, y3], [x1, y1], [x2, y2])
            else:
                [x1, y1] = CalcFourthPoint([x2, y2], [x4, y4], [x3, y3])

        new_polys.append(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))

    return new_polys


def adjust2(ploys):
    temp_boxes = []
    new_boxes = []
    for j in ploys:
        x_min = j[np.lexsort(j[:, ::-1].T)][0][0]		# 排序
        x_max = j[np.lexsort(j[:, ::-1].T)][-1][0]		# 排序
        y_min = j[np.lexsort(j.T)][0][1]
        y_max = j[np.lexsort(j.T)][-1][1]
        new_boxes.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
        temp_box = j[np.lexsort(j.T)]

        if temp_box[0][0] > temp_box[1][0]:
            temp_val = copy.deepcopy(temp_box[0])
            temp_box[0] = temp_box[1]
            temp_box[1] = temp_val

        if temp_box[2][0] < temp_box[3][0]:
            temp_val = copy.deepcopy(temp_box[2])
            temp_box[2] = temp_box[3]
            temp_box[3] = temp_val

        temp_boxes.append(temp_box)

    return temp_boxes


if __name__ == '__main__':
    # polys_1 = np.array([[[20, 10], [20, 20], [10, 20], [10, 10]]])
    polys_1 = np.array([[[15, 5], [20, 10], [15, 25], [10, 10]]])

    ########## adjust test ###########
    # 这段代码感觉有问题！利用平行四边形三个点求第四个点
    adjust1_ploys = adjust(polys_1)
    print('adjust1_ploys: \n', adjust1_ploys)

    ########## adjust2 test ###########
    # 调整坐标点的顺序，使其从左上角开始顺时针输出
    adjust2_ploys = adjust2(polys_1)
    print(adjust2_ploys)