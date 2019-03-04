#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import torch
import numpy as np
from torch.utils.data import DataLoader
from model.C3D_model import C3D
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import models
from dataset_RGB import VideoDataset

device = torch.device('cuda:1')

train_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='train',
    n_frame=16)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=8)

val_data = VideoDataset(
    root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
    split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    split='val',
    n_frame=16)
val_loader = DataLoader(val_data, batch_size=4, shuffle=True, num_workers=8)

n_epoch = 100
lr = 0.01
interval = 500

class RGBModel(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.model = models.vgg16(pretrained=False)
        # self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 7)
        self.fc = nn.Linear(1000, 7)
        # self.model.classifier.add_module('7', nn.Linear(self.model.classifier[6].out_features, self.n_class))

        for name, param in self.model.named_parameters():
            param.requires_grad = True
        for name, param in self.model.classifier.named_parameters():
            param.requires_grad = True

    def forward(self, buf):
        # print('buf shape is {}'.format(buf.shape))
        n_batch = buf.size(0)
        n_frame = buf.size(1)
        res = None

        for idx in range(n_batch):
            output = self.model(buf[idx])
            # print('output shape {}'.format(output.shape))
            output = output.reshape(n_frame, -1)
            # print('output reshape {}'.format(output.shape))
            output = self.fc(output)
            # print('forward', output)
            output = torch.mean(output, 0, keepdim=True)
            # print('forward mean', output)

            if idx == 0:
                res = output
            else:
                res = torch.cat((res, output), 0)
        return res

model = RGBModel(n_class=7)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005 )
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005 )
# optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

def train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, model_dir):
    print('Start trianning')
    RGB_record = open('./RGB_record.txt', 'w+')
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
            # print('model output:\n', outputs)
            # print(preds)

            loss = criterion(preds, labels) * buf.size(0)

            train_loss += loss.item()
            # print('loss:', loss.item())

            _, pred_label = torch.max(preds, 1)

            print('pred label', pred_label)
            print('true label', labels)

            train_corrects += torch.sum(pred_label == labels)

            all_samples = 0
            if (idx+1) %  interval == 0:
                all_samples = (idx+1) * buf.size(0)
                train_loss  = train_loss / all_samples
                print('RGB processing [current:{}/ total:{}],  train_loss  {:.4f}'.format(all_samples, len(train_data), train_loss))
                train_loss = 0

                train_acc = train_corrects.cpu().item() / all_samples
                print('RGB processing train_acc {:.4f}  [{}/{}]'.format(train_acc, train_corrects, all_samples))
            loss.backward()
            optimizer.step()

        train_acc = train_corrects.cpu().item() / len(train_data)
        print('[*] RGB [train-e-{}/{}] [train_acc-{:.4f}, train_loss-{:.4f}][{}/{}]'.
                format(epoch, n_epoch, train_acc, train_loss, train_corrects, len(train_data)))
        with open('./RGB_record.txt', 'a+') as RGB_record:
            RGB_record.write('[train-e-{}/{}] [train_acc-{:.4f} train_loss-{:.4f}] [{}/{}] \n'.
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
        with open('./RGB_record.txt', 'a+') as RGB_record:
            RGB_record.write('[val-e-{}/{}] [val_acc-{:.4f} val_loss-{:.4f}] [{}/{}] \n'.
                    format(epoch, n_epoch, val_acc, val_loss, train_corrects, len(val_data)))
        if val_acc >= 0.70:
            try:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), os.path.join(model_dir,'c3d_momentum_new_{:.4f}.pth'.format(val_acc)))
            except Exception as e:
                print(str(e))
                with open('./RGB_record.txt', 'a+') as RGB_record:
                    RGB_record.write('[ERROR] ' + str(e) + '\n')

if __name__ == '__main__':
    train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, '/home/datasets/mayilong/PycharmProjects/p55/trained_model/rgb')
