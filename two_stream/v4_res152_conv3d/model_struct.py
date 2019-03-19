#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchsummary import summary
from model.model import Model

# device = torch.device('cpu')
device = torch.device('cuda:0')
def test():
    model = Model(7).to(device)
    summary(model, [(3, 224, 224), (1, 10, 224, 224)])

if __name__ == '__main__':
    print('*'*80)
    test()
