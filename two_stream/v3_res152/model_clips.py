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
from sklearn.metrics import classification_report


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
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for idx, (rgb_buf, flow_buf, labels) in enumerate(test_loader):
            rgb_buf = rgb_buf.to(device)
            flow_buf = flow_buf.to(device)
            labels = labels.to(device)
            outputs = model(rgb_buf, flow_buf)
            pred_labels.extend(torch.max(outputs, 1)[1])
            true_labels.extend(labels)
    print(classification_report(true_labels, pred_labels))
    del true_labels
    del pred_labels
    return 

if __name__ == '__main__':
    print('*'*80)
    predict()
