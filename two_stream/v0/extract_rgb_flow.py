#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import cv2 as cv
import pickle

video_data_path = '/home/datasets/mayilong/PycharmProjects/p55/data/split_data'
store_path = '/home/datasets/mayilong/PycharmProjects/p55/data/rgb_flow_300'

def extract_videos(video_data_path):
    video2path = pickle.load(
            open('/home/datasets/mayilong/PycharmProjects/p55/resource/train_val_video2path.pkl', 'rb'))
    for split in os.listdir(video_data_path):

        for txt_file in os.listdir(os.path.join(video_data_path, split)):
            cls = txt_file.split('.')[0]
            print('process {} - {}'.format(split, cls))
            handle = open(os.path.join(video_data_path, split, txt_file), 'r')
            for line in handle.readlines():
                vdo_name = line.strip().split(' ')[0]

                if not os.path.exists(os.path.join(store_path, split, cls)):
                    os.makedirs(os.path.join(store_path, split, cls))

                try:
                    extract_rgb_flow(video2path[vdo_name], os.path.join(os.path.join(store_path, split, cls)))
                except Exception as e:
                    print(e)

def extract_rgb_flow(vdo_path, store_path):
    vdo_name = os.path.basename(vdo_path).split('.')[0]
    err_file = open('./extract_err.txt', 'w+')
    cap = cv.VideoCapture(vdo_path)

    if not os.path.exists(os.path.join(store_path, vdo_name)):
        os.mkdir(os.path.join(store_path, vdo_name))

    all_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(all_frames)
    img_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    idx = 0
    start_idx = np.random.randint(1, all_frames - 40)
    prev_img = None
    prev_img_gray = None
    try:
        while idx < all_frames:
            ret, rgb = cap.read()
            if ret == False:
                cap.release()
                break

            if idx < start_idx - 1:
                idx += 1
                continue
            elif idx == start_idx - 1:
                prev_img = rgb
                cv.imwrite(os.path.join(store_path, vdo_name, 'rgb.jpeg'), rgb)
                prev_img_gray = cv.cvtColor(prev_img, cv.COLOR_RGB2GRAY)

            elif idx >= start_idx and idx < start_idx + 37:
                if (idx - start_idx) % 4 == 0:
                    cur_img = rgb
                    cur_img_gray = cv.cvtColor(cur_img, cv.COLOR_RGB2GRAY)
                    flow = cv.calcOpticalFlowFarneback(prev_img_gray, cur_img_gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)
                    flow_x = flow[..., 0]
                    flow_y = flow[..., 1]
                    cv.imwrite(os.path.join(store_path, vdo_name, 'flowx_{:03d}.jpeg'.format(idx - start_idx)), flow_x)
                    cv.imwrite(os.path.join(store_path, vdo_name, 'flowy_{:03d}.jpeg'.format(idx - start_idx)), flow_y)
                    prev_img_gray = cur_img_gray

            if idx == start_idx + 37:
                cap.release()
                return
            idx += 1
    except Exception as e:
        print(e)
        with open('./extract_err.txt', 'w+') as f:
            f.write('video_name : {}\n'.format(vdo_name)) 
            f.write('err : {}\n'.format(str(e)))
    finally:
        cap.release()

if __name__ == '__main__':
    print('*'*80)
    extract_videos(video_data_path)
