# /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from v1_rgb_flow_dataset import VideoDataset
# from model.flow_resnet import *

device = torch.device('cuda:1')

train_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb_flow_300',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='train')
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

val_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb_flow_300',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='val')
val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=4)

n_epoch = 150
lr = 0.001
interval = 50

class RGBModel_Scratch(nn.Module):
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
        self.model.classifier.add_module('7', nn.Linear(1000, n_class))

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
        outputs = self.model(buf)
        return outputs

class FlowModel(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.model = models.vgg16(pretrained=True)

        print('change first layer parameter')
        old_param = self.model.features[0].weight.data
        old_param = torch.mean(old_param, dim=1, keepdim=True)
        new_param = old_param.repeat(1, 20, 1, 1)
        self.model.features[0] = nn.Conv2d(20, 64, 3, 1,1)
        self.model.features[0].weight.data = new_param
        self.model.classifier.add_module('7', nn.Linear(1000, n_class))

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        print('load flow model finish')

    def forward(self, buf):
        outputs = self.model(buf)
        return outputs

rgb_model = RGBModel(n_class=7)
rgb_model = rgb_model.to(device)
flow_model = FlowModel(n_class=7)
flow_model =  flow_model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD([{'params':rgb_model.model.classifier.parameters()}, {'params':flow_model.model.classifier.parameters(), 'lr':0.0001}], lr=lr, momentum=0.9, weight_decay=0.0005 )
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005 )
# optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2, last_epoch=-1)

def train_model(rgb_model, flow_model, n_epoch, optimizer, scheduler, train_loader, val_loader, model_dir):
    print('Start trainning')
    record = open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]), 'w+')
    for epoch in range(n_epoch):
        rgb_model.train()
        flow_model.train()
        corrects = 0
        rgb_loss, flow_loss = 0, 0
        rgb_corrects, flow_corrects = 0, 0
        total = 0

        for idx, (rgb_buf, flow_buf, labels) in enumerate(train_loader):

            #buf = buf.to(device)
            rgb_buf = rgb_buf.to(device)
            flow_buf = flow_buf.to(device)
            labels = labels.to(device)
            # print('t rbuf: ', rgb_buf.size())
            # print('t fbuf: ', flow_buf.size())
            # print('t label: ', labels.size())

            rgb_outputs = rgb_model(rgb_buf)
            flow_outputs = flow_model(flow_buf)
            outputs = 0.2*flow_outputs + 0.8*rgb_outputs

            batch_rgb_loss = criterion(rgb_outputs, labels)
            batch_flow_loss = criterion(flow_outputs, labels)
            loss = 0.3*batch_rgb_loss + 0.7*batch_flow_loss

            _, pred_labels = torch.max(outputs, 1)

            # accumulate correct samples, loss and all samples 
            corrects += torch.sum(pred_labels == labels).item()
            rgb_corrects += torch.sum(torch.max(rgb_outputs, 1)[1] == labels).item()
            flow_corrects += torch.sum(torch.max(flow_outputs, 1)[1] == labels).item()

            rgb_loss += batch_rgb_loss.item()
            flow_loss += batch_flow_loss.item()
            print('rgb_loss:{}, flow_loss:{}'.format(rgb_loss, flow_loss))
            total += rgb_buf.size(0)

            # print('pred label', pred_labels)
            # print('true label', labels)

            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        rgb_loss = rgb_loss / total
        flow_loss = flow_loss / total
        acc = corrects / total
        with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]),  'a+') as record:
            record.write('[rgb_acc-{:.4f}, flow_acc-{:.4f}]\n'.format(rgb_corrects / total, flow_corrects / total))
            record.write('[train-{}/{}] [acc-{:.4f}, loss-{:.4f}] [cor/total-{}/{}]\n'.format(
                    epoch, n_epoch, acc, rgb_loss+flow_loss, corrects, total))

        rgb_model.eval()
        flow_model.eval()
        with torch.no_grad():
            rgb_loss, flow_loss = 0, 0
            corrects = 0
            total = 0
            rgb_corrects, flow_corrects = 0, 0

            for idx, (rgb_buf, flow_buf, labels) in enumerate(val_loader):
                rgb_buf = rgb_buf.to(device)
                flow_buf = flow_buf.to(device)
                labels = labels.to(device)
                # print('v rbuf: ', rgb_buf.size())
                # print('v fbuf: ', flow_buf.size())
                # print('v label: ', labels.size())

                rgb_outputs = rgb_model(rgb_buf)
                flow_outputs = flow_model(flow_buf)

                batch_rgb_loss = criterion(rgb_outputs, labels)
                batch_flow_loss = criterion(flow_outputs, labels)
                loss = 0.5*batch_rgb_loss + 0.5*batch_flow_loss

                rgb_loss += batch_rgb_loss.item()
                flow_loss += batch_flow_loss.item()

                outputs = 0.5*rgb_outputs + 0.5*flow_outputs
                _, pred_labels = torch.max(outputs, 1)
                corrects += torch.sum(pred_labels == labels).item()
                total += rgb_buf.size(0)

                rgb_corrects += torch.sum(torch.max(rgb_outputs, 1)[1] == labels).item()
                flow_corrects += torch.sum(torch.max(flow_outputs, 1)[1] == labels).item()
            # may modify learning rate
            rgb_loss  = rgb_loss / total
            flow_loss  = flow_loss / total
            scheduler.step(loss)
            acc = corrects / total


            with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]),  'a+') as record:
                record.write('[v:rgb_acc-{:.4f}, flow_acc-{:.4f}]\n'.format(rgb_corrects / total, flow_corrects / total))
                record.write('[val-{}/{}] [acc-{:.4f}, loss-{:.4f}] [cor/total-{}/{}]\n'.format(
                    epoch, n_epoch, acc, rgb_loss+flow_loss, corrects, total))

            if acc >= 0.50:
                try:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)

                    torch.save(rgb_model.state_dict(), os.path.join(model_dir,'rgb_flow_vgg16_{:.4f}.pth'.format(acc)))
                except Exception as e:
                    print(str(e))
                    with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]),  'a+') as record:
                        record.write('[ERROR] ' + str(e) + '\n')

if __name__ == '__main__':
    train_model(rgb_model, flow_model, n_epoch, optimizer, scheduler, train_loader, val_loader, '/home/datasets/mayilong/PycharmProjects/p55/two_stream/trained_model/')
