#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import models
from dataset import VideoDataset

device = torch.device('cuda:1')

train_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='train',
    n_frame=16)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)

val_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='val',
    n_frame=16)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=8)

n_epoch = 150
lr = 0.0001
interval = 20

class RGBModelS(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.classfiler = nn.Sequential(
                nn.Linear(25088, 1024),
                nn.Linear(1024,7))

        for name, m in self.features.named_modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.normal_(m.bias, 0, 1)
                print(name, ' init!')

        for name, m in self.classfiler.named_modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.normal_(m.bias, 0, 1)
                print(name, ' init!')

    def forward(self, buf):
        output = self.features(buf)
        output = output.view(buf.size(0), -1)
        output = self.classfiler(output)
        return output

class RGBModel(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.model = models.vgg16(pretrained=True)
        self.model.classifier.add_module('7', nn.Linear(1000, 7))

        # for name, m in self.model.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.normal_(m.weight)
        #         torch.nn.init.normal_(m.bias)
        #         print(name, ' init!')

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.model.classifier.named_parameters():
            param.requires_grad = True

    def forward(self, buf):
        n_batch = buf.size(0)
        res = None
        for i in range(n_batch):
            output = self.model(buf[i])
            output, _ = torch.max(output, dim=0, keepdim=True)
            if i == 0:
                res = output
            else:
                res = torch.cat((output, res), 0)
        return res

model = RGBModel(n_class=7)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005 )
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005 )
# optimizer = optim.SGD(filter(lambda m: m.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1, -1)

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

def train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, model_dir):
    print('Start trianning')
    record = open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]), 'w+')
    for epoch in range(n_epoch):
        model.train()
        corrects = 0
        total = 0
        total_loss = 0
        loss = 0

        for idx, (buf, labels) in enumerate(train_loader):

            buf = buf.to(device)
            labels = labels.to(device)
            outputs = model(buf)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, pred_label = torch.max(outputs, 1)
            corrects += torch.sum(pred_label == labels).item()
            total += buf.size(0)

            print('pred label', pred_label)
            print('true label', labels)

            if (idx+1) %  interval == 0:
                print('[train-{}/{}] [{}/{}] [acc-{:.4f} loss-{:.4f}]'.format(epoch, n_epoch, corrects, total, corrects/total, total_loss/total))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('[train-{}/{}] [{}/{}] [acc-{:.4f} loss-{:.4f}]'.format(epoch, n_epoch, corrects, total, corrects/total, total_loss/total))
        with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]),  'a+') as record:
            record.write('[train-{}/{}] [{}/{}] [acc-{:.4f} loss-{:.4f}]'.format(epoch, n_epoch, corrects, total, corrects/total, total_loss/total))

        model.eval()
        with torch.no_grad():
            corrects = 0
            total = 0
            loss = 0
            total_loss = 0
            acc = 0

            for idx, (buf, labels) in enumerate(val_loader):
                buf = buf.to(device)
                labels = labels.to(device)
                outputs = model(buf)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                total += buf.size(0)
                _, pred_labels = torch.max(preds, 1)
                corrects += torch.sum(pred_labels == labels).item()

            # may modify learning rate
            scheduler.step(loss)
            print('[val-{}/{}] [{}/{}] [acc-{:.4f} loss-{:.4f}]'.format(epoch, n_epoch, corrects, total, corrects/total, total_loss/total))

            with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]),  'a+') as record:
                record.write('[val-{}/{}] [{}/{}] [acc-{:.4f} loss-{:.4f}]'.
                        format(epoch, n_epoch, corrects, total, corrects/total, total_loss/total))

            if acc >= 0.50:
                try:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)

                    torch.save(model.state_dict(), os.path.join(model_dir,'vgg16_16_{:.4f}.pth'.format(corrects/total)))
                except Exception as e:
                    print(str(e))
                    with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]),  'a+') as record:
                        record.write('[ERROR] ' + str(e) + '\n')

if __name__ == '__main__':
    train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, os.path.join(os.getcwd(), 'trained_model'))
