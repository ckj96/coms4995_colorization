"""
help to sample images
@author: ciciphus
"""

import os
import re
import argparse
import random
import numpy as np



def list_pictures(directory, ext='jpg'):
    """
    List of images in the directory.

    Args:
        directory: directory that contains images

    Returns:
        list of image names
    """

    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


class Sampler(object):

    def __init__(self, root_dir, pos_num, neg_num, output_dir):
        self.root_dir = root_dir
        self.dir_lev1 = os.listdir(root_dir)
        self.dir_lev2 = []
        self.pos_num = pos_num
        self.neg_num = neg_num
        self.output_dir = output_dir

        for directory in self.dir_lev1:
            self.dir_lev2.append(os.listdir(os.path.join(root_dir, directory)))

        # save the length of directory
        self.dict_dir = dict()
        self.cul_len = [0]
        cnt = 0
        self.dir2images = dict()

        for index, dir2 in enumerate(self.dir_lev2):
            self.cul_len.append(self.cul_len[-1] + len(dir2))

            for i in range(len(dir2)):
                path = os.path.join(self.root_dir, self.dir_lev1[index], dir2[i])
                imgs = list_pictures(path)
                self.dir2images[dir2[i]] = imgs
                self.dict_dir[cnt] = index
                cnt += 1

        self.num_cat = self.cul_len[-1]
        print('sum_catorgory:', self.cul_len)
        self.triplets = []

    def get_dir(self, num):
        index1 = self.dict_dir[num]
        dir1 = self.dir_lev1[index1]

        index2 = num - self.cul_len[index1]
        dir2 = (self.dir_lev2[index1])[index2]
        return os.path.join(self.root_dir, dir1, dir2)

    def get_images(self, directory):
        directory = directory.split('/')[-1]
        return self.dir2images[directory]

    def random_num(self, num):
        return min(int(random.random() * num), num - 1)

    # return absolute path
    def get_random_sample(self, directory, num):
        imgs = []
        path = self.get_images(directory)
        for i in range(num):
            rand = self.random_num(len(path))
            # imgs.append(os.path.join(directory, path[rand]))
            imgs.append(path[rand])
        return imgs

    def generate_neg(self, dir2):
        neg_imgs = list()
        i = 0
        nums = set()
        while i < self.neg_num:
            while True:
                num = self.random_num(self.cul_len[-1])
                dir_neg = self.get_dir(num)
                if dir_neg != dir2:
                    break

            imgs = self.get_images(dir_neg)
            index_img = self.random_num(len(imgs))

            if num not in nums:
                nums.add(num)
                neg_imgs.append((imgs[index_img], num))
                i += 1
        return neg_imgs

    def dict2str(self, query_img, pos_dict, pos_root, query_index):
        query_index = str(query_index)
        for (pos, negs) in pos_dict.items():
            for (neg, neg_num) in negs:
                triplet = query_img + ',' + os.path.join(pos_root, pos) + ',' + neg + ',' + query_index + \
                            ',' + query_index + ',' + str(neg_num) + '\n'
                self.triplets.append(triplet)

    # note: dir2 is absolute path, r
    def generate_triplets_in_cat(self, query_imgs_path, dir2, query_index):

        images = self.get_images(dir2)
        for query_img_path in query_imgs_path:
            pos_images_dict = dict()
            query_img = query_img_path
            for i in range(self.pos_num):
                while True:
                    index = self.random_num(len(images))
                    img = images[index]
                    if query_img != img and img not in pos_images_dict:
                        neg_imgs = self.generate_neg(dir2)
                        pos_images_dict[img] = neg_imgs
                        break
            self.dict2str(query_img_path, pos_images_dict, dir2, query_index )

    def generate_triplets(self, num):
        num_each_cat = num // self.num_cat
        for dir_index in range(self.num_cat):
            directory = self.get_dir(dir_index)
            print(directory)
            query_imgs_path = self.get_random_sample(directory, num_each_cat)
            self.generate_triplets_in_cat(query_imgs_path, directory, dir_index)
            print(len(self.triplets))

        random.shuffle(self.triplets)

    def write(self):
        print("==> Sampling Done ... Now Writing ...")
        print(len(self.triplets))
        f = open(os.path.join(self.output_dir, "triplets.txt"), 'w')

        for triplet in self.triplets:
            f.write(triplet)
        f.close()


def main():
    """Triplet Sampling."""
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Triplet Sampler arguments')

    parser.add_argument('--input_directory', type=str,
                        default="/home/cc4192/data/data/vision/torralba/deeplearning/images256/",
                        help='input directory')

    parser.add_argument('--output_directory', type=str, default="../",
                        help='output directory')

    parser.add_argument('--num_query_images', type=int, default=300000)

    parser.add_argument('--num_pos_images', type=int, default=2,
                        help='the number of Positive images per Query image')

    parser.add_argument('--num_neg_images', type=int, default=2,
                        help='the number of Negative images per Query image')

    args = parser.parse_args()

    if int(args.num_neg_images) < 1:
        print('Number of Negative Images cannot be less than 1!')
    elif int(args.num_pos_images) < 1:
        print('Number of Positive Images cannot be less than 1!')

    if not os.path.exists(args.input_directory):
        print(args.input_directory + " path does not exist!")
        quit()

    if not os.path.exists(args.output_directory):
        print(args.input_directory + " path does not exist!")
        quit()

    print("Input Directory: " + args.input_directory)
    print("Output Directory: " + args.output_directory)
    print("Number of Positive image per Query image:", args.num_pos_images)
    print("Number of Negative image per Query image:", args.num_neg_images)

    sampler = Sampler(args.input_directory, args.num_pos_images, args.num_neg_images, args.output_directory)
    sampler.generate_triplets(args.num_query_images)
    sampler.write()


if __name__ == '__main__':
    main()
