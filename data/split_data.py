#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import sys
sys.path.append('../')
import os
import numpy as np
from utils.store_utils import parse_pkl
import cv2 as cv
import argparse
import shutil

def check_frames(video, low, high):
    cap = cv.VideoCapture(video)
    frame_len = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    return True if frame_len >= low and frame_len < high else False


def split_train_val_test(cls_txt, save_dir, video2path, low, high, label, n_video=300, train_ratio=0.6, val_ratio=0.8):
    """
    cls_txt : 所有文件的分类文件, 根据此文件挑选数据
    save_dir ：生成的挑选数据txt的目录, 根据挑选的数据进行训练
    video2path : 记录视频和其路径的字典
    """
    cls = os.path.basename(cls_txt)
    print('processing ', cls)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(os.path.join(save_dir, 'train')):
        os.makedirs(os.path.join(save_dir, 'train'))
    if not os.path.exists(os.path.join(save_dir, 'val')):
        os.makedirs(os.path.join(save_dir, 'val'))
    if not os.path.exists(os.path.join(save_dir, 'test')):
        os.makedirs(os.path.join(save_dir, 'test'))

    video2path = parse_pkl(video2path)

    train_txt = os.path.join(os.path.join(save_dir, 'train'), cls)
    val_txt = os.path.join(os.path.join(save_dir, 'val'), cls)
    test_txt = os.path.join(os.path.join(save_dir, 'test'), cls)

    train_handle = open(train_txt, 'w+')
    val_handle = open(val_txt, 'w+')
    test_handle = open(test_txt, 'w+')

    dataset = []
    with open(cls_txt, 'r+') as f:
        for line in f.readlines():
            line = line.strip().split(' ')[0]
            dataset.append(line)

    # 保证不出现重复数据
    dataset = np.random.permutation(dataset)

    count = 0
    for idx, item in enumerate(dataset):
        if count < 300:
            if check_frames(video2path[item], low, high):
                if count < int(n_video*train_ratio):
                    train_handle.write(item + ' ' + str(label) + '\n')
                elif count < int(n_video*val_ratio):
                    val_handle.write(item + ' ' + str(label) + '\n')
                else:
                    test_handle.write(item + ' ' + str(label) + '\n')
                count += 1
        else:
            break
    if count < 300:
        print('{} can not find enough sample'.format(cls))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--video2path', required=True)
    args = parser.parse_args()

    classes = ['cat', 'dance_related', 'dog', 'female_selfie', 'game', 'male_selfie', 'play_related', ]
    labels = [0, 1, 2, 3, 4, 5, 6]

    for cls, label in zip(classes, labels):
        split_train_val_test(os.path.join(args.txt_dir,cls + '.txt'), args.save_dir, args.video2path, 180, 451, label)

