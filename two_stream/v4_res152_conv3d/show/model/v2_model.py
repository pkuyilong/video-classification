#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device('cuda:2')

class Merge():
    def __init__(self):
        super().__init__()

    def merge(self, buf1, buf2):
        if buf1.size() == buf2.size():
            outputs = torch.cat((buf1, buf2), dim=1)
            return outputs
        b1, c1, h1, w1 = buf1.size()
        b2, c2, h2, w2 = buf2.size()

        if b1 != b2:
            raise RuntimeError('batch size not match')

        if h1 == w1 and h2 == w2:
            if h1 < h2:
                min_buf = buf1
                max_buf = buf2
            else:
                min_buf = buf2
                max_buf = buf1
            gap = abs(h1 - h2)
            if gap <= 2:
                if gap == 2:
                    min_buf = nn.ReplicationPad2d((1, 1, 1, 1))(min_buf)
                else:
                    min_buf =  nn.ReplicationPad2d((0, 1, 0, 1))(min_buf)

            elif max_buf.size(2) / min_buf.size()[2] == 2.0:
                    max_buf = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))(max_buf)

            else:
                raise RuntimeError('Cant do anything')

            outputs = torch.cat((min_buf, max_buf), dim=1)
            return outputs

class RGBExtrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        for name, child in models.resnet152(pretrained=True).named_children():
            if type(child).__name__ == 'Linear' or type(child).__name__ == 'AvgPool2d':
                continue
            self.model.add_module(name, child)
        self.model.add_module('conv_1x1', nn.Conv2d(2048, 384, kernel_size=(1,1), stride=(1,1)))

    def forward(self, buf):
        outputs = self.model(buf)
        return outputs

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_a = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(2,3,3), stride=(2,3,3)),
                nn.Conv3d(64, 512, kernel_size=(3,3,3), stride=(1,3,3)),
                nn.Conv3d(512, 128, kernel_size=(3,3,3), stride=(3,3,3))
                )

        self.conv_b = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5,3,3), stride=(5,3,3)),
                nn.Conv3d(64, 512, kernel_size=(2,3,3), stride=(1,3,3)),
                nn.Conv3d(512, 128, kernel_size=(1, 3, 3), stride=(1,3,3))
                )

        self.conv_c = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(10,3,3), stride=(10,3,3)),
                nn.Conv3d(64, 512, kernel_size=(1, 3, 3), stride=(1, 3, 3)),
                nn.Conv3d(512, 128, kernel_size=(1, 3, 3), stride=(1, 3, 3))
                )

    def forward(self, img):
        output_a = self.conv_a(img)
        output_b = self.conv_b(img)
        output_c = self.conv_c(img)
        features = torch.cat((output_a, output_b, output_c), dim=1)
        features = features.squeeze(dim=2)
        return features

class Model(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.rgb_extractor = RGBExtrator()
        self.flow_extractor = Block()
        self.merger = Merge()

        self.fc1 = nn.Linear(2*2*12288,  1024)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, n_class)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, rgb_buf, flow_buf):
        rgb_features = self.rgb_extractor(rgb_buf)
        flow_features = self.flow_extractor(flow_buf)
        features = self.merger.merge(rgb_features, flow_features)
        features = features.view(features.size(0), 4*12288)

        outputs = self.fc1(features)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        return outputs

if __name__ == '__main__':
    print('*'*80)
    img = torch.randn(1, 3, 224, 224).to(device)
    rgb_extractor = RGBExtrator().to(device)
    outputs = rgb_extractor(img)
    print(outputs.size())

    flow = torch.randn(1, 1, 10, 224,224).to(device)
    block = Block().to(device)
    outputs = block(flow)
    print(outputs.size())

    model = Model(7).to(device)
    outputs = model(img, flow)
    print(outputs.size())

