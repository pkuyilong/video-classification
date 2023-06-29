#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from v1_several_dataset import VideoDataset

device = torch.device('cuda:0')

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

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=True)

n_epoch = 100
lr = 0.0001
interval = 500


class RGBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier.add_module(1000, 512)
        self.model.classifier.add_module(512, 7)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.model.classifier.named_parameters():
            param.requires_grad = True

    def forward(self, buf):
        n_frame = buf.size(0)
        res = None
        for idx in range(n_frame):
            output = self.model(buf[idx])
            output, _ = torch.max(output, 0, keepdim=True)
            if idx == 0:
                res = output
            else:
                res = torch.cat((res, output), 0)
        return res


model = RGBModel()
model = model.to(device)
print('prepare model')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)


# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

def train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, model_dir):
    record = open('./{}.txt'.foramt(os.path.basename(__file__).split('.')[0]), 'w')
    for epoch in range(n_epoch):
        model.train()
        corrects = 0
        loss = 0
        total = 0

        for idx, (buf, labels) in enumerate(loader):
            buf = buf.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(buf)

            loss = criterion(outputs, labels)

            loss += loss.item()

            _, pred_label = torch.max(preds, 1)
            print('pred label', pred_label)
            print('true label', labels)

            corrects += torch.sum(pred_label == labels).item()

            if (idx + 1) % interval == 0:
                total += buf.size(0)
                loss = loss / total
                print('RGB processing [current:{}/ total:{}],  loss  {:.4f}'.format(total, total, loss))

                acc = corrects / total
                print('RGB processing acc {:.4f}  [{}/{}]'.format(acc, corrects, total))
            loss.backward()
            optimizer.step()

        acc = corrects / total
        print('[*] RGB [train-e-{}/{}] [acc-{:.4f}, loss-{:.4f}][{}/{}]'.format(epoch, n_epoch, acc, loss, corrects,
                                                                                total))
        with open('./{}.txt'.foramt(os.path.basename(__file__).split('.')[0]), 'a+') as record:
            record.write(
                '[train-e-{}/{}] [acc-{:.4f} loss-{:.4f}] [{}/{}] \n'.format(epoch, n_epoch, acc, loss, corrects,
                                                                             total))

        model.eval()
        with torch.no_grad():
            corrects = 0
            loss = 0
            total = 0

            for idx, (buf, labels) in enumerate(loader):
                optimizer.zero_grad()

                buf = buf.to(device)
                labels = labels.to(device)

                outputs = model(buf)

                loss = criterion(outputs, labels)
                loss += loss.item()

                _, pred_labels = torch.max(preds, 1)
                corrects += torch.sum(pred_labels == labels).item()
                total += buf.size(0)

            loss = loss / total
            # scheduler.step(loss)
            acc = corrects / total
            print('[val-e-{}/{}] [{}/{}]'.format(epoch, n_epoch, corrects, total))
            print('acc {:.4f}, loss {:.4f}'.format(acc, loss))
            with open('./{}.txt'.foramt(os.path.basename(__file__).split('.')[0]), 'a+') as record:
                record.write('[val-e-{}/{}] [acc-{:.4f} loss-{:.4f}] [{}/{}] \n'.
                             format(epoch, n_epoch, acc, loss, corrects, total))

            # whether save model
            if acc >= 0.70:
                try:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    torch.save(model.state_dict(), os.path.join(model_dir, 'RGB_densenet_1_{:.4f}.pth'.format(acc)))
                except Exception as e:
                    print(str(e))
                    with open('./record.txt', 'a+') as record:
                        record.write('[ERROR] ' + str(e) + '\n')


if __name__ == '__main__':
    train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader,
                '/home/datasets/mayilong/PycharmProjects/p55/trained_model/rgb')
