#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import os
import pickle

def gen_split_data_label(split_data_root,  rgb_root, split, store_pos):
    """
    make a dict to match split_data and its label based on unsplit data-label file

    split_data_root: folder contrain train_data, test_data, val_data , eacho folder contrain cls.txt
    rgb_root: folder , contrains train, val , test
    split : train or val or test
    """
    unsplit_video2label = dict()
    for txt_file in os.listdir(os.path.join(split_data_root, split+'_data')):
        # print(os.path.join(split_data_root, split+'_data', txt_file))
        file_handle = open(os.path.join(split_data_root, split+'_data', txt_file), 'r+')
        for line in file_handle.readlines():
            unsplit_video2label[line.split(' ')[0].strip()] = line.split(' ')[1].strip()

    split_video2label = dict()
    # video_name = xxxxx.mp4_x 
    for cls_folder in os.listdir(os.path.join(rgb_root, split)):
        for video_name in os.listdir(os.path.join(rgb_root, split, cls_folder)):
            name = video_name[:video_name.rfind('_')]
            split_video2label[video_name] = unsplit_video2label[name]

    with open(os.path.join(store_pos, '{}_video2label.pkl'.format(split)), 'wb') as f:
        pickle.dump(split_video2label, f)

if __name__ == '__main__':
    print('*'*80)

    for split in ['train', 'val', 'test']:
        print('processing --', split)
        gen_split_data_label('/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
                '/home/datasets/mayilong/PycharmProjects/p55/data/rgb/',
                split,
                '/home/datasets/mayilong/PycharmProjects/p55/resource')

