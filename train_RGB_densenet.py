#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
from dataset_RGB import VideoDataset

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
lr = 0.01
interval = 500

class RGBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 7)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.model.classifier.named_parameters():
            param.requires_grad = True

    def forward(self, buf):
        n_frame = buf.size(0)
        res = None
        for idx in range(n_frame):
            output = self.model(buf[idx])
            output = nn.Softmax(dim=1)(output)
            output, _ = torch.max(output, 0)
            output = output.unsqueeze(0)
            if idx == 0:
                res = output
            else:
                res = torch.cat((res, output), 0)
        return res

model = RGBModel()
model = model.to(device)
print('prepare model')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005 )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

def train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, model_dir):
    record = open('./{}.txt'.foramt(os.path.basename(__file__).split('.')[0]), 'w')
    for epoch in range(n_epoch):
        model.train()
        train_corrects = 0
        train_loss = 0
        train_total = 0

        for idx, (buf, labels) in enumerate(train_loader):
            buf = buf.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(buf)


            loss = criterion(outputs, labels) * buf.size(0)

            train_loss += loss.item()

            _, pred_label = torch.max(preds, 1)
            print('pred label', pred_label)
            print('true label', labels)

            train_corrects += torch.sum(pred_label == labels).item()

            if (idx+1) %  interval == 0:
                train_total += buf.size(0)
                train_loss  = train_loss / train_total
                print('RGB processing [current:{}/ total:{}],  train_loss  {:.4f}'.format(train_total, train_total, train_loss))

                train_acc = train_corrects / train_total
                print('RGB processing train_acc {:.4f}  [{}/{}]'.format(train_acc, train_corrects, train_total))
            loss.backward()
            optimizer.step()

        train_acc = train_corrects / train_total
        print('[*] RGB [train-e-{}/{}] [train_acc-{:.4f}, train_loss-{:.4f}][{}/{}]'.
                format(epoch, n_epoch, train_acc, train_loss, train_corrects, train_total))
        with open('./{}.txt'.foramt(os.path.basename(__file__).split('.')[0]), 'a+') as record:
            record.write('[train-e-{}/{}] [train_acc-{:.4f} train_loss-{:.4f}] [{}/{}] \n'.
                    format(epoch, n_epoch, train_acc, train_loss, train_corrects, train_total))

        model.eval()
        with torch.no_grad():
            val_corrects = 0
            val_loss = 0
            val_total = 0

            for idx, (buf, labels) in enumerate(val_loader):
                optimizer.zero_grad()

                buf = buf.to(device)
                labels = labels.to(device)

                outputs = model(buf)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, pred_labels = torch.max(preds, 1)
                val_corrects += torch.sum(pred_labels == labels).item()
                val_total += buf.size(0)

            val_loss = val_loss / val_total

            # may update lr
            # scheduler.step(val_loss)

            val_acc = val_corrects / val_total
            print('[val-e-{}/{}] [{}/{}]'.format(epoch, n_epoch, val_corrects, val_total))
            print('val_acc {:.4f}, val_loss {:.4f}'.format(val_acc, val_loss))
            with open('./{}.txt'.foramt(os.path.basename(__file__).split('.')[0]), 'a+') as record:
                record.write('[val-e-{}/{}] [val_acc-{:.4f} val_loss-{:.4f}] [{}/{}] \n'.
                    format(epoch, n_epoch, val_acc, val_loss, val_corrects, val_total))

            # whether save model
            if val_acc >= 0.70:
                try:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    torch.save(model.state_dict(), os.path.join(model_dir,'RGB_densenet_1_{:.4f}.pth'.format(val_acc)))
                except Exception as e:
                    print(str(e))
                    with open('./record.txt', 'a+') as record:
                        record.write('[ERROR] ' + str(e) + '\n')

if __name__ == '__main__':
    train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, '/home/datasets/mayilong/PycharmProjects/p55/trained_model/rgb')
