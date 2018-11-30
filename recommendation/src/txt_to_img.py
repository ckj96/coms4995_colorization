import os
import argparse
import shutil
from shutil import copyfile

def save_single(txt_path, output_path):
    txt_name = txt_path.split('/')[-1]
    txt_name = txt_name.split('.')[0]

    output_img_dir = os.path.join(output_path, txt_name)
    if os.path.isdir(output_img_dir):
        shutil.rmtree(output_img_dir)
    os.mkdir(output_img_dir)

    for line in open(txt_path):
        line = line.strip('\n')
        img_name = line.split('/')[-1]
        img_path = os.path.join(output_img_dir, img_name)
        if args.cp_file:
            copyfile(line, img_path)
        else:
            os.symlink(line, img_path)


def save_mul(dir_path, output_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        save_single(file_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--txt_path', type=str,
                        default="../", help='path to txt')

    parser.add_argument('--directory_path', type=str, default=None, help='path to directory save multiple txt' )
    parser.add_argument('--output_path', type=str, default=None, help='where to store the result')
    parser.add_argument('--cp_file', type=bool, default=False, help='copy file or create link')
    args = parser.parse_args()

    if args.directory_path:
        save_mul(args.directory_path, args.output_path)
    else:
        save_single(args.txt_path, args.output_path)