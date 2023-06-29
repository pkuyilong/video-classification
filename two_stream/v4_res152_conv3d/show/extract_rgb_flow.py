#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import os
import pickle

import cv2 as cv
import numpy as np


def handle_rgb_buf(rgb_buf, save_dir):
    # usage rgb_buf (list contain 11 rgb frame)  save_dir (path/to/video_name)
    if len(rgb_buf) == 0 or len(rgb_buf) != 11:
        raise RuntimeError('rgb_buf is empty')
    rgb_file = rgb_buf[5]
    cv.imwrite(os.path.join(save_dir, 'rgb.jpeg'), rgb_file)
    prev_gray = cv.cvtColor(rgb_buf[0], cv.COLOR_RGB2GRAY)
    for i in range(1, len(rgb_buf)):
        cur_gray = cv.cvtColor(rgb_buf[i], cv.COLOR_RGB2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        cv.imwrite(os.path.join(save_dir, 'flowx_{:02d}.jpeg'.format(i - 1)), flow_x)
        cv.imwrite(os.path.join(save_dir, 'flowy_{:02d}.jpeg'.format(i - 1)), flow_y)
        prev_gray = cur_gray
        cur_gray = None


def rgb_flow_gen(video, video2path, save_dir):
    # usage :
    # rgb_flow_gen('928296508.mp4', video2path, '/home/datasets/mayilong/PycharmProjects/p55/two_stream/v1/data/video_tmp')
    vdo_path = video2path[video]
    vdo_name = video
    cap = cv.VideoCapture(vdo_path)
    all_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    img_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    idx = 0
    interval = np.random.randint(0, 60)
    suffix = 0
    start_pos = np.random.randint(10, 60)
    prev_gray = None
    cur_gray = None
    already_save_rgb = False
    rgb_buf = []
    need_break = False
    while idx < all_frames:
        ret, frame = cap.read()
        if ret is False:
            rgb_buf.clear()
            cap.release()

        if idx < start_pos:
            idx += 1
            continue

        if need_break:
            for i in range(interval):
                ret, frame = cap.read()
                if ret is False:
                    cap.release()
                    break
            need_break = False
            interval = np.random.randint(0, 60)

        if (idx - start_pos) % 4 == 0:
            rgb_buf.append(frame)
            if len(rgb_buf) == 11:
                rgb_of_folder = os.path.join(os.path.join(save_dir, vdo_name + '_' + str(suffix)))
                if not os.path.exists(rgb_of_folder):
                    os.mkdir(rgb_of_folder)
                suffix += 1
                handle_rgb_buf(rgb_buf, rgb_of_folder)
                rgb_buf.clear()
                need_break = True
                if suffix >= 4:
                    cap.release()
                    break


if __name__ == '__main__':
    video2path = pickle.load(
        open('/home/datasets/mayilong/PycharmProjects/p55/resource/train_val_video2path.pkl', 'rb'))
    # video_name = '976443169.mp4'
    # video_name = '956328232.mp4'
    # video_name = '944844406.mp4'
    # video_name = '986706308.mp4' #  female
    # video_name = '898450942.mp4'
    video_name = '946235928.mp4'  # play_related
    video2path[video_name]
    print(video2path[video_name])

    rgb_flow_gen(video_name, video2path, os.getcwd())

    # extractor.run()
