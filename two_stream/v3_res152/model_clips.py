#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from model.model import Model
from dataset import VideoDataset
from torch.utils.data import DataLoader
import time

device = torch.device('cuda:2')

root_dir = '/home/datasets/mayilong/data_warehouse/two_stream_data/dataset3/data'
split_data = '/home/datasets/mayilong/data_warehouse/two_stream_data/dataset3/split_data'

test_data = VideoDataset(
    root_dir=root_dir,
    split_data=split_data,
    split='test',
    )
test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=4)

model = Model(7).to(device)
model.load_state_dict(torch.load('./trained_model/two_stream_0.8521.pth'))
print('load model success')

def predict():
    corrects_so_far = 0
    count_so_far = 0
    print('Start trainning')
    begin_time = time.time()
    test_size = len(test_data)

    pred_list = [0 for i in range(7)]
    true_list = [0 for i in range(7)]
    model.eval()
    with torch.no_grad():
        for idx, (rgb_buf, flow_buf, labels) in enumerate(test_loader):
            rgb_buf = rgb_buf.to(device)
            flow_buf = flow_buf.to(device)
            labels = labels.to(device)
            outputs = model(rgb_buf, flow_buf)
            pred_labels = torch.max(outputs, 1)[1]

            # print('t ', labels)
            # print('p ', pred_labels)

            corrects_so_far += torch.sum(pred_labels == labels).item()
            count_so_far += rgb_buf.size(0)
            if (idx + 1) % 100 == 0:
                print('[acc:{:.4f} {}/{}]'.format(corrects_so_far/count_so_far, corrects_so_far, count_so_far))

            labels = labels.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
            for i, j in zip(pred_labels, labels):
                true_list[j] += 1
                if i == j:
                    pred_list[i] +=1

        print('[final acc:{:.4f} {}/{}]'.format(corrects_so_far/count_so_far,corrects_so_far, count_so_far))
        cost_time = time.time() - begin_time
        print(cost_time)
        print(cost_time / test_size)
        print(pred_list)
        print(true_list)
        print(np.array(pred_list) / np.array(true_list))

if __name__ == '__main__':
    print('*'*80)
    predict()
