#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
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

        if c1 != c2 and h1 == h2 and w1 == w2:
            outputs = torch.cat((buf1, buf2), dim=1)
            return outputs

        if c1 == c2:
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

                assert min_buf.size() == max_buf.size()

                outputs = torch.cat((min_buf, max_buf), dim=1)
                print(outputs.size())
                return outputs


class RGBExtrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        for name, child in models.resnet152(pretrained=True).named_children():
            if type(child).__name__ == 'Linear' or type(child).__name__ == 'AvgPool2d':
                continue
            self.model.add_module(name, child)

    def forward(self, buf):
        outputs = self.model(buf)
        return outputs

class FlowExtrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        for name, child in models.resnet152(pretrained=True).named_children():
            if type(child).__name__ == 'Linear' or type(child).__name__ == 'AvgPool2d':
                continue
            self.model.add_module(name, child)
        old_param = self.model.conv1.weight.data
        old_param_mean = torch.mean(old_param, dim=1, keepdim=True)
        new_param = old_param_mean.repeat(1, 20, 1, 1)
        self.model.conv1 = nn.Conv2d(20, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.conv1.weight.data = new_param

    def forward(self, buf):
        outputs = self.model(buf)
        return outputs

class Model(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.rgb_extractor = RGBExtrator()
        self.flow_extractor = FlowExtrator()
        self.merger = Merge()

        self.conv1 = nn.Conv2d(4096, 512, 3, 1)
        self.conv2 = nn.Conv2d(512, 32, 3, 1)
        self.fc = nn.Linear(32*3*3, n_class)

    def forward(self, rgb_buf, flow_buf):
        rgb_features = self.rgb_extractor(rgb_buf)
        flow_features = self.flow_extractor(flow_buf)
        # features = torch.cat((rgb_features, flow_features), dim=1)
        features = self.merger.merge(rgb_features, flow_features)
        outputs = self.conv1(features)
        outputs = self.conv2(outputs)
        outputs = outputs.view(-1, 32*3*3)
        outputs = self.fc(outputs)
        return outputs

if __name__ == '__main__':
    print('*'*80)
    img = torch.randn(1, 3, 224, 224)
    rgb_extractor = RGBExtrator()
    outputs = rgb_extractor(img)
    print(outputs.size())

    flow = torch.randn(1, 20, 224,224)
    flow_extractor = FlowExtrator()
    outputs = flow_extractor(flow)
    print(outputs.size())

    model = Model(7)
    outputs = model(img, flow)
    print(outputs.size())



