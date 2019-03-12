#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.C3D_model import C3D
import torch.nn as nn
import torch.optim as optim
from dataset import VideoDataset

device = torch.device('cuda:1')

train_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='train',
    n_frame=16)
val_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='val',
    n_frame=16)

train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=True)

n_epoch = 300
lr = 0.00001
interval = 50

# model = C3D(num_classes=7, pretrained=True)
# for module in model.modules():
#     if isinstance(module, nn.Conv3d):
#         module.weight.requires_grad = False
#         module.bias.requires_grad = False
#     elif isinstance(module, nn.Linear):
#         module.weight.requires_grad = True
#         module.bias.requires_grad = True
#     else:
#         continue

# load trained model

model = C3D(num_classes=7, pretrained=True)
# model = C3D(num_classes=7, pretrained=False)
# model.load_state_dict(torch.load('./trained_model/c3d_new_0.7200.pth'))
# print('load trained model by myself')

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005 )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

def train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, model_dir):
    print('Start trianning')
    record = open(os.path.join(os.getcwd(), '{}.txt'.format(os.path.basename(__file__).split('.')[0])), 'w+')
    for epoch in range(n_epoch):
        model.train()
        corrects = 0
        total_loss = 0
        total = 0
        loss = 0

        for idx, (buf, labels) in enumerate(train_loader):
            buf = buf.to(device)
            labels = labels.to(device)
            outputs = model(buf)

            loss = criterion(outputs, labels)
            _, pred_label = torch.max(outputs, 1)

            total_loss += loss.item()
            corrects += torch.sum(pred_label == labels).item()
            total += buf.size(0)

            print('pred label', pred_label)
            print('true label', labels)

            if (idx+1) %  interval == 0:
                print('[acc-{:.4f}, loss-{:.4f} [{}/{}]'.format(corrects/total, total_loss/total, corrects, total))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('[train-{}/{}] [acc-{:.4f}, loss-{:.4f}] [{}/{}]\n'.format(epoch, n_epoch, corrects/total, total_loss/total, corrects, total))
        with open(os.path.join(os.getcwd(), '{}.txt'.format(os.path.basename(__file__).split('.')[0])), 'a+') as record:
            record.write('[train-{}/{}] [acc-{:.4f}, loss-{:.4f}] [{}/{}]\n'.
                    format(epoch, n_epoch, corrects/total, total_loss/total, corrects, total))

        model.eval()
        with torch.no_grad():
            corrects = 0
            total = 0
            total_loss = 0

            for idx, (buf, labels) in enumerate(val_loader):
                buf = buf.to(device)
                labels = labels.to(device)
                outputs = model(buf)

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total += buf.size(0)
                _, pred_labels = torch.max(outputs, 1)
                corrects += torch.sum(pred_labels == labels).item()

            # may modify learning rate
            scheduler.step(loss)
            print('[val-{}/{}] [acc-{:.4f}, loss-{:.4f}] [{}/{}]\n'.format(epoch, n_epoch, corrects/total, total_loss/total, corrects, total))

            with open(os.path.join(os.getcwd(), '{}.txt'.format(os.path.basename(__file__).split('.')[0])), 'a+') as record:
                record.write('[val-{}/{}] [acc-{:.4f}, loss-{:.4f}] [{}/{}]\n'.
                        format(epoch, n_epoch, corrects/total, total_loss/total, corrects, total))

            if corrects/total >= 0.74:
                try:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)

                    torch.save(model.state_dict(), os.path.join(model_dir,'rgb_stream_{:.4f}.pth'.format(corrects/total)))

                except Exception as e:
                    print(str(e))
                    with open(os.path.join(os.getcwd(), '{}.txt'.format(os.path.basename(__file__).split('.')[0])), 'a+') as record:
                        record.write('[ERROR] ' + str(e) + '\n')

if __name__ == '__main__':
    train_model(model,
                n_epoch,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                os.path.join(os.getcwd(), 'trained_model'))

