#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

from torchsummary import summary

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


device = torch.device('cuda:0')
class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_a = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(2,3,3), stride=(2,3,3)),
                nn.Conv3d(64, 512, kernel_size=(3,3,3), stride=(1,3,3)),
                nn.Conv3d(512, 64, kernel_size=(3,3,3), stride=(3,3,3))
                )

        self.conv_b = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5,3,3), stride=(5,3,3)),
                nn.Conv3d(64, 512, kernel_size=(2,3,3), stride=(1,3,3)),
                nn.Conv3d(512, 64, kernel_size=(1, 3, 3), stride=(1,3,3))
                )

        self.conv_c = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(10,3,3), stride=(10,3,3)),
                nn.Conv3d(64, 512, kernel_size=(1, 3, 3), stride=(1, 3, 3)),
                nn.Conv3d(512, 64, kernel_size=(1, 3, 3), stride=(1, 3, 3))
                )
        self.classifier = nn.Sequential(
                nn.Linear(64*3*8*8, 8192),
                nn.Linear(8192, 512),
                nn.Linear(512, 7)
                )

    def forward(self, img):
        output_a = self.conv_a(img)
        output_b = self.conv_b(img)
        output_c = self.conv_c(img)
        features = torch.cat((output_a, output_b, output_c), dim=1)
        features = features.view(img.size(0), -1)
        outputs= self.classifier(features)
        return outputs

if __name__ == '__main__':
    print('*'*80)
    img = torch.randn(1, 1, 10, 224, 224).to(device)
    b = Block().to(device)
    summary(b, (1, 10, 224, 224))
    # print(b(img).size())

