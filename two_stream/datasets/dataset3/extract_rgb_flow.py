#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import os
import pickle

import cv2 as cv
import numpy as np


class Extractor():
    def __init__(self, cls_root, save_dir, video2path):
        super().__init__()
        self.cls_root = cls_root
        self.save_dir = save_dir
        self.video2path = video2path
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.record = open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]), 'w+')

    def run(self):
        for split in os.listdir(self.cls_root):
            if split == 'test' or split == 'val':
                continue
            if not os.path.exists(os.path.join(self.save_dir, split)):
                os.mkdir(os.path.join(self.save_dir, split))

            for cls_txt in os.listdir(os.path.join(self.cls_root, split)):
                cls = cls_txt.split('.')[0]
                print('processing split-{}, cls-{}'.format(split, cls))
                if not os.path.join(self.save_dir, split, cls):
                    os.mkdir(os.path.join(save_dir, split, cls))

                for idx, line in enumerate(open(os.path.join(cls_root, split, cls_txt), 'r+').readlines()):
                    print(idx)
                    vdo_name = line.split(' ')[0]
                    if not os.path.exists(os.path.join(self.save_dir, split, cls)):
                        os.mkdir(os.path.join(self.save_dir, split, cls))
                    try:
                        self.rgb_flow_gen(vdo_name, os.path.join(self.save_dir, split, cls))
                    except Exception as e:
                        print(e)
                        with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0], 'a+')) as record:
                            record.write('[Error] {}\n'.format(str(e)))

    def rgb_flow_gen(self, video, save_dir):
        # usage :
        # rgb_flow_gen('928296508.mp4', '/home/datasets/mayilong/PycharmProjects/p55/two_stream/v1/data/video_tmp')
        vdo_name = video.split('.')[0]
        vdo_path = video2path[video]
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
                    self.handle_rgb_buf(rgb_buf, rgb_of_folder)
                    rgb_buf.clear()
                    need_break = True
                    if suffix >= 4:
                        cap.release()
                        break

            idx += 1

    def handle_rgb_buf(self, rgb_buf, save_dir):
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


if __name__ == '__main__':
    # rgb_flow_gen('928296508.mp4', '/home/datasets/mayilong/PycharmProjects/p55/two_stream/v1/data/video_tmp')
    cls_root = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/v1/data/split_data'
    save_dir = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/v1/data/datasets'
    video2path = pickle.load(
        open('/home/datasets/mayilong/PycharmProjects/p55/resource/train_val_video2path.pkl', 'rb'))
    extractor = Extractor(cls_root, save_dir, video2path)
    extractor.run()
