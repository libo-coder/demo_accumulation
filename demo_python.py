# -*- coding: utf-8 -*-
"""
一些常用的 Python 代码合计整理
@author: libo
"""

def all_unique(lst):
    """ 检查给定的列表是不是存在重复元素  使用 set() 函数来移除所有重复元素 """
    return len(lst) == len(set(lst))


def most_frequent(lst):
    """ 根据元素频率取列表中最常见的元素 """
    return max(set(lst), key=lst.count)


from collections import Counter
def anagram(first, second):
    """ 检查两个字符串的组成元素是不是一样的 """
    return Counter(first) == Counter(second)


def byte_size(string):
    """ 检查字符串占用的字节数 """
    return len(string.encode('utf-8'))


from math import ceil
def chunk(lst, size):
    """ 分块：给定具体的大小，定义一个函数以按照这个大小切割列表 """
    return list(map(lambda x: lst[x * size, x * size + size], list(range(0, ceil(len(lst) / size)))))


def compact(lst):
    """ 压缩：将布尔值过滤掉  使用 filter() 函数 """
    return list(filter(bool, lst))


def unzip(array):
    """ 解包：将打包好的成对列表解开成两组不同的元组 """
    transposed = zip(*array)
    return transposed

# array = [['a', 'b'], ['c', 'd'], ['e', 'f']]
# print(unzip(array))         # [('a', 'c', 'e'), ('b', 'd', 'e')]


def spread(arg):
    ret = []
    for i in arg:
        if isinstance(i, list):
            ret.extend(i)
        else:
            ret.append(i)
    return ret

def deep_flatten(lst):
    """ 通过递归的方式将列表的嵌套展开为单个列表 """
    result = []
    result.extend(spread(list(map(lambda x: deep_flatten(x) if type(x) == list else x, lst))))
    return result

# deep_flatten([1, [2], [[3], 4], 5])     # [1, 2, 3, 4, 5]


def different(a, b):
    """ 列表的差：返回第一个列表的元素，其不在第二个列表内 """
    set_a = set(a)
    set_b = set(b)
    comparison = set_a.difference(set_b)
    return comparison


def to_dictionary(keys, values):
    """ 将两个列表转化为字典 """
    return dict(zip(keys, values))


def palindrome(string):
    """ 回文序列 """
    from re import sub
    s = sub('[\W_]', '', string.lower())
    return s == s[::-1]


####### 利用自带的缓存机制提高效率：缓存是一种将定量数据加以保存，以备迎合后续获取需求的处理方式，旨在加快数据获取的速度。
from functools import lru_cache
import timeit

@lru_cache(None)
def fib(n):
    if n < 2:
        return n
    return fib(n - 2) + fib(n - 1)

# print(timeit.timeit(lambda: fib(500), number=1))
##############################################################


################# 流式读取数G超大文件 ##########################
"""
使用 with...open... 可以从一个文件中读取数据，但是当使用了 read 函数，其实 Python 会将文件的内容一次性的全部载入内存中，
如果文件有 10G 甚至更多，那么电脑就要消耗的内存非常巨大。
"""
# 一次性读取
def read_from_file0(filename):
    with open(filename, "r") as fp:
        content = fp.read()
    return content

# 使用 readline 去做一个生成器来逐行返回。
# 可如果这个文件内容就一行呢，一行就 10个G，其实你还是会一次性读取全部内容。
def read_from_file1(filename):
    with open(filename, "r") as fp:
        yield fp.readline()

# 在使用 read 方法时，指定每次只读取固定大小的内容
def read_from_file2(filename, block_size=1024 * 8):
    with open(filename, "r") as fp:
        while True:
            chunk = fp.read(block_size)
            if not chunk:
                break
            yield chunk

# 优化：借助偏函数 和 iter 函数优化一下代码
from functools import partial
def read_from_file3(filename, block_size=1024 * 8):
    with open(filename, "r") as fp:
        for chunk in iter(partial(fp.read, block_size), ""):
            yield chunk
##############################################################


################### 使用 print 输出日志 ########################
def write_log(filename):
    with open(filename, mode='w') as f:
        print('hello, python', file=f, flush=True)

# write_log('./log_test.txt')
##############################################################


################### 将嵌套 for 循环写成单行 #####################
from itertools import product

def iter_test():
    list1 = range(1, 3)
    list2 = range(4, 6)
    list3 = range(7, 9)
    for item1, item2, item3 in product(list1, list2, list3):
        print(item1 + item2 + item3)

# iter_test()
##############################################################