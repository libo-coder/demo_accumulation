# coding=utf-8
"""
工作目录中文件后缀名修改
运行指令：
    python change_file_ext.py --work_dir=./test/ --old_ext=jpg --new_ext=png

@author: libo
"""
import os
import argparse     # 定义脚本参数

def get_parser():
    parser = argparse.ArgumentParser(description='change_file_ext')
    parser.add_argument('--work_dir', default='', type=str, nargs=1, help='WORK_DIR')
    parser.add_argument('--old_ext', default='', type=str, nargs=1, help='OLD_EXT')
    parser.add_argument('--new_ext', default='', type=str, nargs=1, help='NEW_EXT')
    return parser


def batch_rename(work_dir, old_ext, new_ext):
    """ 后缀名批量修改 """
    for filename in os.listdir(work_dir):
        bfn, ext = os.path.splitext(filename)       # 获取得到文件后缀
        # 定位后缀名为old_ext 的文件
        if old_ext == ext:
            newfile = bfn + new_ext     # 修改后文件的完整名称
            os.rename(os.path.join(work_dir, filename), os.path.join(work_dir, newfile))    # 实现重命名操作
            print("完成重命名")
            print(os.listdir(work_dir))


def main():
    # 命令行参数
    parser = get_parser()
    print(parser)
    args = vars(parser.parse_args())
    # 从命令行参数中依次解析出参数
    work_dir = args['work_dir'][0]
    print(work_dir)

    old_ext = args['old_ext'][0]
    if old_ext[0] != '.':
        old_ext = '.' + old_ext

    new_ext = args['new_ext'][0]
    if new_ext[0] != '.':
        new_ext = '.' + new_ext

    batch_rename(work_dir, old_ext, new_ext)


if __name__ == '__main__':
    main()