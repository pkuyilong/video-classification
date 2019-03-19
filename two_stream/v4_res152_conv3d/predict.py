#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from model.v2_model import Model

device = torch.device('cuda:0')

dataset_path = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/datasets/dataset3/data'
split_data = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/dataset/split_data'

test_data = VideoDataset(
    dataset_path=dataset_path,
    split_data=split_data,
    split='test',
    )
test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=4)

model = Model(7).to(device)
model.load_state_dict(torch.load('./trained_model/xxx')['state_dict'])


def predict():
    corrects_so_far = 0
    count_so_far = 0
    for idx, (rgb_buf, flow_buf, labels) in enumerate(test_loader):
        outputs = model(rgb_buf, flow_buf)
        pred_labels = torch.max(outputs, 1)[1]
        corrects_so_far += torch.sum(pred_labels == labels)
        count_so_far += rgb_buf.size(0)
        if (idx + 1) % 100 == 0:
            print('[acc:{:.4f} {}/{}]'.format(corrects_so_far/count_so_far,
                corrects_so_far, count_so_far)

    print('[final acc:{:.4f} {}/{}]'.format(corrects_so_far/count_so_far,
            corrects_so_far, count_so_far)


if __name__ == '__main__':
    print('*'*80)
