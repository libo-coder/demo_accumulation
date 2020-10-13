# coding=utf-8
"""
批量修改文件名
运行指令：
    python change_file_name.py --selected_images=./test/ --images_new_path=./test2/
@author: libo
"""
import os
import shutil
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='change_file_name')
    # parser.add_argument('--work_dir', default='', type=str, nargs=1, help='WORK_DIR')
    parser.add_argument('--selected_images', default='', type=str, nargs=1, help='image_path')
    parser.add_argument('--images_new_path', default='', type=str, nargs=1, help='image_new_path')
    return parser


def batch_rename(old_path, new_path):
    """ 批量修改文件名 """
    for img_name in os.listdir(old_path):
        print(img_name)
        bfn, ext = os.path.splitext(img_name)
        if ext.lower() not in ['.jpg', '.png']:
            continue
        new_name = bfn + '2'
        shutil.copyfile(os.path.join(old_path, img_name), os.path.join(new_path, new_name + ext))
        print("完成重命名")


def main():
    # 命令行参数
    parser = get_parser()
    args = vars(parser.parse_args())
    # work_dir = args['work_dir'][0]
    old_path = args['selected_images'][0]
    new_path = args['images_new_path'][0]

    batch_rename(old_path, new_path)


if __name__ == '__main__':
    main()