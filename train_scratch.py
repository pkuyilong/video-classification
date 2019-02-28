#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 mayilong <mayilong@img>
#
# Distributed under terms of the MIT license.

import torch
import numpy as np
from dataset import VideoDataset
from torch.utils.data import DataLoader
from model.C3D_model import C3D
from model.C3D_scratch import C3D_S
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device('cuda:0')

# train_data  = VideoDataset(root_dir='/home/datasets/mayilong/PycharmProjects/p44/data/rgb', split='train')
train_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='train',
    n_frame=16)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# val_data  = VideoDataset(root_dir='/home/datasets/mayilong/PycharmProjects/p44/data/rgb', split='val')
val_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='val',
    n_frame=16)
val_loader = DataLoader(val_data, batch_size=4, shuffle=True)

test_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='test',
    n_frame=16)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

n_epoch = 300
lr = 0.0001
interval = 500

model = C3D(num_classes=7, pretrained=True)
for module in model.modules():
    if isinstance(module, nn.Conv3d):
        module.weight.requires_grad = True
        module.bias.requires_grad = True
    elif isinstance(module, nn.Linear):
        module.weight.requires_grad = True
        module.bias.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
optimizer = optim.SGD(model.parameters(), lr=lr)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

def train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, model_dir):
    record = open('./record.txt', 'w+')
    for epoch in range(n_epoch):
        model.train()
        train_corrects = 0
        train_loss = 0
        for idx, (buf, labels) in enumerate(train_loader):
            buf = buf.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(buf)

            preds = nn.Softmax(dim=1)(outputs)

            loss = criterion(preds, labels) * buf.size(0)

            train_loss += loss.item()

            _, pred_label = torch.max(preds, 1)

            # print('pred label', pred_label)
            # print('true label', labels)

            train_corrects += torch.sum(pred_label == labels)

            all_samples = 0
            if (idx+1) %  interval == 0:
                all_samples = (idx+1) * buf.size(0)
                train_loss  = train_loss / all_samples
                print('processing [current:{}/ total:{}],  train_loss  {:.4f}'.format(all_samples, len(train_data), train_loss))
                train_loss = 0

                train_acc = train_corrects.cpu().item() / all_samples
                print('processing train_acc {:.4f}  [{}/{}]'.format(train_acc, train_corrects, all_samples))
            loss.backward()
            optimizer.step()

        train_acc = train_corrects.cpu().item() / len(train_data)
        print('[*] [train-e-{}/{}] [train_acc-{:.4f}, train_loss-{:.4f}][{}/{}]'.
                format(epoch, n_epoch, train_acc, train_loss, train_corrects, len(train_data)))
        with open('./record.txt', 'a+') as record:
            record.write('[train-e-{}/{}] [train_acc-{:.4f} train_loss-{:.4f}] [{}/{}] \n'.
                    format(epoch, n_epoch, train_acc, train_loss, train_corrects, len(train_data)))

        model.eval()
        val_corrects = 0
        val_loss = 0
        for idx, (buf, labels) in enumerate(val_loader):
            optimizer.zero_grad()

            buf = buf.to(device)
            labels = labels.to(device)

            outputs = model(buf)

            preds = nn.Softmax(dim=1)(outputs)

            _, pred_labels = torch.max(preds, 1)

            loss = criterion(preds, labels) * buf.size(0)
            val_loss += loss.item()

            val_corrects += torch.sum(pred_labels == labels)

        val_loss  = val_loss / len(val_data)
        scheduler.step(val_loss)
        val_acc = val_corrects.cpu().item() / len(val_data)
        print('[val-e-{}/{}] [{}/{}]'.format(epoch, n_epoch, val_corrects, len(val_data)))
        print('val_acc {:.4f}, val_loss {:.4f}'.format(val_acc, val_loss))
        with open('./record.txt', 'a+') as record:
            record.write('[val-e-{}/{}] [val_acc-{:.4f} val_loss-{:.4f}] [{}/{}] \n'.
                    format(epoch, n_epoch, val_acc, val_loss, train_corrects, len(val_data)))
        if val_acc >= 0.70:
            try:
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                torch.save(model.state_dict(), os.path.join(model_dir,'c3d_new_{:.4f}.pth'.format(val_acc)))
            except Exception as e:
                print(str(e))
                with open('./record.txt', 'a+') as record:
                    record.write('[ERROR] ' + str(e) + '\n')

if __name__ == '__main__':
    train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, '/home/datasets/mayilong/PycharmProjects/p55/trained_model')
