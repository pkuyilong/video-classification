#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import VideoDataset
from model.C3D_model import C3D
from torch.utils.data import DataLoader

device = torch.device('cuda:2')
test_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='test',
    n_frame=16)

test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

model = C3D(7)

def test():
    model.load_state_dict(torch.load('./trained_model/c3d_new_0.7226.pth'))
    model.to(device)

    test_corrects = 0
    for idx, (buf, labels) in enumerate(test_loader):
        buf = buf.to(device)
        labels = labels.to(device)
        outputs = model(buf)
        preds = torch.max(outputs, 1)[1]
        print('p :', preds)
        print('t :', labels)
        test_corrects += torch.sum(preds == labels).item()

    print('test_acc-{}'.format(test_corrects / len(test_loader))) 

if __name__ == '__main__':
    print('*'*80)
    test()
