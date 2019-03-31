#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from model.model import Model
from torch.utils.data import DataLoader
import cv2 as cv

device = torch.device('cuda:0')

model_dir = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/v4_res152_conv3d/trained_model/two_stream_0.8575.pth'
dataset_path = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/datasets/dataset3/data'
split_data = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/dataset/split_data'

model = Model(7).to(device)
model.load_state_dict(torch.load(model_dir)['state_dict'])
model = model

resize_height = 280
resize_width = 280
crop_size = 224
def load_frames(video_folder):
    # return 1 rgb and 20 optical flow
    rgb_buf = None
    for name in os.listdir(video_folder):
        if name.startswith('rgb'):
            rgb_buf = cv.imread(os.path.join(video_folder,name)).astype(np.float32)
            rgb_buf = cv.resize(rgb_buf, (resize_height, resize_width))
            rgb_buf[..., 0] = rgb_buf[..., 0] - np.average(rgb_buf[..., 0])
            rgb_buf[..., 1] = rgb_buf[..., 1] - np.average(rgb_buf[..., 1])
            rgb_buf[..., 2] = rgb_buf[..., 2] - np.average(rgb_buf[..., 2])
            start_height = np.random.randint(0, rgb_buf.shape[0] - crop_size + 1)
            start_width = np.random.randint(0, rgb_buf.shape[1] - crop_size + 1)
            rgb_buf = rgb_buf[start_height : start_height+crop_size,
                    start_width : start_width+crop_size, :]
            rgb_buf = rgb_buf.transpose(2, 0, 1)
    if rgb_buf is None:
        raise ValueError('not found any rgb images')

    flowx_files = sorted([os.path.join(video_folder, name) for name in os.listdir(video_folder) if name.startswith('flowx')])
    flowy_files = sorted([os.path.join(video_folder, name) for name in os.listdir(video_folder) if name.startswith('flowy')])

    # check flowx and flowy  image
    cur_flowx_num = len(flowx_files)
    if cur_flowx_num == 0:
        print(video_folder, 'has no flowx frame')
    while cur_flowx_num < 10:
        print(video_folder, cur_flowx_num)
        need_to_fill = 10 - cur_flowx_num
        while need_to_fill > 0:
            flowx_files.append(flowx_files[-1])
            need_to_fill -= 1

    cur_flowy_num = len(flowy_files)
    if cur_flowy_num == 0:
        print(video_folder, 'has no flowy frame')
    while cur_flowy_num < 10:
        print(video_folder, cur_flowy_num)
        need_to_fill = 10 - cur_flowy_num
        while need_to_fill > 0:
            flowy_files.append(flowy_files[-1])
            need_to_fill -= 1

    flow_buf = np.empty((resize_height, resize_width, 10), np.dtype('float32'))

    for idx, (flowx, flowy) in enumerate(zip(flowx_files, flowy_files)):
        flow_x = cv.imread(flowx, 0).astype(np.float32)
        flow_y = cv.imread(flowy, 0).astype(np.float32)
        flow_x = cv.resize(flow_x, (resize_width, resize_height))
        flow_y = cv.resize(flow_y, (resize_width, resize_height))

        flow = np.max((flow_x, flow_y), axis=0)
        flow_buf[:, :, idx] = flow

    start_height = np.random.randint(0, flow_buf.shape[0] - crop_size + 1)
    start_width = np.random.randint(0, flow_buf.shape[1] - crop_size + 1)
    flow_buf = flow_buf[start_height : start_height+crop_size, start_width : start_width+crop_size, :]
    flow_buf = flow_buf.transpose(2, 0, 1)

    flow_buf = flow_buf[np.newaxis, ...]
    return (rgb_buf, flow_buf)

def predict(rgb_buf, flow_buf):
    corrects_so_far = 0
    count_so_far = 0
    rgb_buf = rgb_buf.to(device)
    flow_buf = flow_buf.to(device)
    outputs = model(rgb_buf, flow_buf)
    pred_labels = torch.max(outputs, 1)[1]
    print('predicted labels ', pred_labels)

if __name__ == '__main__':
    print('*'*80)

    # rgb_buf, flow_buf = load_frames('./986706308.mp4_2')
    # rgb_buf, flow_buf = load_frames('898450942.mp4_2')
    for i in range(3):
        rgb_buf, flow_buf = load_frames('./946235928.mp4_' + str(i))
        rgb_buf = rgb_buf[np.newaxis, :, :, :]
        flow_buf = flow_buf[np.newaxis, :, :, :, :]
        rgb_buf = torch.from_numpy(rgb_buf)
        flow_buf = torch.from_numpy(flow_buf)
        predict(rgb_buf, flow_buf)
