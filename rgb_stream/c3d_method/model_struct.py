#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from model.C3D_model import C3D
from torchsummary import summary




device = torch.device('cpu')
# device = torch.device('cuda:0')
def test():
    c3d_model = C3D(7, pretrained=False).to(device)
    summary(c3d_model, input_size=(3, 16, 112, 112))



if __name__ == '__main__':
    print('*'*80)
    test()
