#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device('cuda:1')

class RGB_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 7)

    def forward(self, buf):
        output = self.model(buf)
        output, _ = torch.max(nn.Softmax()(output), 0)
        print(output)
        return output

def test():
    imgs = torch.randn(16, 3, 224, 224).to(device)
    model = RGB_model().to(device)
    output = model(imgs)
    print(output.shape)

if __name__ == '__main__':
    print('*'*80)
    test()
