# coding=utf-8
"""
matplotlib画图的常用技巧
@author: libo
"""
import matplotlib.pyplot as plt
import numpy as np

def add_textbox():
    """ 为文本添加背景框 """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # fig = plt.figure()
    fig, axe = plt.subplots()
    axe.text(0.5, 0.5, '我是文本框', bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 0.7})
    plt.show()


def add_annotate():
    """ 添加指示箭头 """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, axe = plt.subplots()
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2 * np.pi * t)
    axe.plot(t, s, linestyle='-', label='line1')
    axe.annotate('我是正弦函数', xy=(1.25, 1), xytext=(1.9, 1),
                 arrowprops=dict(facecolor='red', shrink=0.2),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.show()


def bar_horizontal():
    """ 柱状图横置 """
    fig, axe = plt.subplots()
    data_m = (40, 60, 120, 180, 20, 200)
    index = np.arange(6)
    width = 0.4
    axe.barh(index, data_m, width, align='center', alpha=0.8, label='men')
    plt.show()


def bar_style():
    fig, axe = plt.subplots()
    data_m = (40, 60, 120, 180, 20, 200)
    index = np.arange(6)
    # axe.bar(index, data_m, color='y')   # 改变图形颜色
    axe.bar(index, data_m)
    plt.style.use('dark_background')
    plt.show()


def change_scale():
    """ 改变刻度 """
    fig, axe = plt.subplots()
    axe.set_xticks([0, 1, 2, 3, 4, 5])
    axe.set_xticklabels(['Taxi', 'Metro', 'Walk', 'Bus', 'Bicycle', 'Driving'])
    # axe.set_xticklabels(['Taxi', 'Metro', 'Walk', 'Bus', 'Bicycle', 'Driving'], rotation=45)        # 坐标倾斜
    plt.show()


def sub_plot():
    """ 绘制子图 """
    fig, axe = plt.subplots(4, 4, figsize=(10, 10))
    plt.show()


def grid_figure():
    """ 绘制带网格线的图 """
    fig, axe = plt.subplots()
    axe.grid(True)
    plt.show()


if __name__ == '__main__':
    # add_textbox()
    # add_annotate()
    # bar_horizontal()
    # change_scale()
    # sub_plot()
    # grid_figure()
    bar_style()