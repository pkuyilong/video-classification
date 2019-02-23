#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 mayilong <mayilong@img>
#
# Distributed under terms of the MIT license.

"""

"""
import pickle
import os

def generate_video2label(root_dir, split):
    d = dict()
    for label, txt_file in enumerate(sorted(os.listdir(root_dir))):
        txt_path = os.path.join(root_dir, txt_file)
        handle = open(txt_path, 'r')
        for video in handle.readlines():
            video = video.strip()
            d[video] = label

    with open('./resource/{}_video2label.pkl'.format(split), 'wb') as f:
        pickle.dump(d, f)
    # for idx, (k,v) in enumerate(d.items()):
    #     if idx < 10:
    #         print(k)
    #         print(v)


if __name__ == '__main__':
    generate_video2label('/home/datasets/mayilong/PycharmProjects/p44/data/train_data', 'train')
    generate_video2label('/home/datasets/mayilong/PycharmProjects/p44/data/val_data', 'val')
    generate_video2label('/home/datasets/mayilong/PycharmProjects/p44/data/test_data', 'test')

