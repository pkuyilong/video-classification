#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import sys

sys.path.append('./model')
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import VideoDataset
from model.flow_resnet import *

device = torch.device('cuda:1')

# train_data = VideoDataset(
#     root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
#     split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
#     split='train',
#     n_frame=16)
# train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1)
# 
# val_data = VideoDataset(
#     root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
#     split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
#     split='val',
#     n_frame=16)
# val_loader = DataLoader(val_data, batch_size=8, shuffle=True, num_workers=8)

train_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb_flow_300',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='train',
    ty='flow',
    n_frame=20)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

val_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb_flow_300',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='val',
    ty='flow',
    n_frame=20)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=4)

n_epoch = 150
lr = 0.0001
interval = 50


class RGBModel2(nn.Module):
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
            nn.Linear(1024, 7))

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
        # self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 7)
        # self.fc = nn.Linear(1000, 7)
        # self.model.classifier.add_module('7', nn.Linear(self.model.classifier[6].out_features, self.n_class))

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
        output = self.model(buf)
        return output


class FlowModel(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.model = models.vgg16(pretrained=True)

        print('load model parameter')
        old_param = self.model.features[0].weight.data
        old_param = torch.mean(old_param, dim=1, keepdim=True)
        new_param = old_param.repeat(1, 20, 1, 1)
        self.model.features[0] = nn.Conv2d(20, 64, 3, 1, 1)
        self.model.features[0].weight.data = new_param

        # self.model.load_state_dict(torch.load('./pretrained_model/ucf101_s1_flow_resnet152.pth.tar')['state_dict'])
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(self.model.classifier[6].out_features, 7)
        self.dp = nn.Dropout(0.3)

    def forward(self, buf):
        output = self.dp(self.model(buf))
        output = self.fc(output)
        return output


# model = RGBModel(n_class=7)
# model = model.to(device)
model = FlowModel(n_class=7)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005)
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
        train_corrects = 0
        train_total = 0
        train_loss = 0

        for idx, (buf, labels) in enumerate(train_loader):
            buf = buf.to(device)
            labels = labels.to(device)
            outputs = model(buf)

            loss = criterion(outputs, labels)
            print('train loss ', loss)
            train_loss += loss.item()
            train_total += buf.size(0)

            _, pred_label = torch.max(outputs, 1)
            train_corrects += torch.sum(pred_label == labels).item()

            # print('pred label', pred_label)
            # print('true label', labels)

            # if (idx+1) %  interval == 0:
            #     train_loss  = train_loss / train_total
            #     print('RGB processing [current:{}/ total:{}],  train_loss  {:.4f}'.format(train_total, len(train_data), train_loss))

            #     train_acc = train_corrects / train_total
            #     print('RGB processing train_acc {:.4f}  [{}/{}]'.format(train_acc, train_corrects, train_total))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = train_corrects / train_total
        train_loss = train_loss / train_total
        # print('[*] RGB [train-e-{}/{}] [train_acc-{:.4f}, train_loss-{:.4f}][{}/{}]'.
        #         format(epoch, n_epoch, train_acc, train_loss, train_corrects, train_total))

        with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]), 'a+') as record:
            record.write('[train-e-{}/{}] [train_acc-{:.4f} train_loss-{:.4f}] [{}/{}] \n'.
                         format(epoch, n_epoch, train_acc, train_loss, train_corrects, train_total))

        model.eval()
        with torch.no_grad():
            val_corrects = 0
            val_total = 0
            val_loss = 0

            for idx, (buf, labels) in enumerate(val_loader):
                buf = buf.to(device)
                labels = labels.to(device)
                outputs = model(buf)
                loss = criterion(outputs, labels)
                print('val loss ', loss)
                val_loss += loss.item()
                val_total += buf.size(0)

                _, pred_labels = torch.max(outputs, 1)
                val_corrects += torch.sum(pred_labels == labels).item()

            # may modify learning rate
            val_loss = val_loss / val_total
            scheduler.step(val_loss)
            val_acc = val_corrects / val_total

            with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]), 'a+') as record:
                record.write('[val-e-{}/{}] [val_acc-{:.4f} val_loss-{:.4f}] [{}/{}]\n'.
                             format(epoch, n_epoch, val_acc, val_loss, val_corrects, val_total))
            # print('[val-e-{}/{}] [{}/{}]'.format(epoch, n_epoch, val_corrects, val_total))
            # print('val_acc {:.4f}, val_loss {:.4f}'.format(val_acc, val_loss))

            if val_acc >= 0.50:
                try:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)

                    torch.save(model.state_dict(),
                               os.path.join(model_dir, 'flow_pre_resnet152_{:.4f}.pth'.format(val_acc)))
                except Exception as e:
                    print(str(e))
                    with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]), 'a+') as record:
                        record.write('[ERROR] ' + str(e) + '\n')


if __name__ == '__main__':
    train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader,
                '/home/datasets/mayilong/PycharmProjects/p55/two_stream/trained_model/flow')
