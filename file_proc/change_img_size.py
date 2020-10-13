# coding=utf-8
"""
批量修改文件夹下的图片大小
运行指令：
    python change_img_size.py --work_dir=./test/ --out_dir=./resize_test/

@author: libo
"""
import os
import cv2
import argparse     # 定义脚本参数

def get_parser():
    parser = argparse.ArgumentParser(description='change_img_size')
    parser.add_argument('--work_dir', default='', type=str, nargs=1, help='WORK_DIR')
    parser.add_argument('--out_dir', default='', type=str, nargs=1, help='OUT_DIR')
    return parser


def batch_resize(work_dir, out_dir, width=128, height=32):
    """ 图片大小批量修改 """
    for imgname in os.listdir(work_dir):
        bfn, ext = os.path.splitext(imgname)
        if ext.lower() not in ['.jpg', '.png', '.tiff']:
            continue
        img_src = cv2.imread(work_dir + imgname, cv2.IMREAD_ANYCOLOR)
        try:
            img_dst = cv2.resize(img_src, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(out_dir, imgname), img_dst)
        except Exception as e:
            print(e)


def main():
    parser = get_parser()
    args = vars(parser.parse_args())
    work_dir = args['work_dir'][0]
    out_dir = args['out_dir'][0]
    batch_resize(work_dir, out_dir)


if __name__ == '__main__':
    main()