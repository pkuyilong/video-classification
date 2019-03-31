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
from dataset import VideoDataset
from model.model import Model

device = torch.device('cuda:2')

dataset_path = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/datasets/dataset3/data'
split_data = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/dataset/split_data'

train_data = VideoDataset(
    dataset_path=dataset_path,
    split_data=split_data,
    split='train',
    )
val_data = VideoDataset(
    dataset_path=dataset_path,
    split_data=split_data,
    split='val',
    )

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=4)


model = Model(7)
model = model.to(device)

n_epoch = 1000
lr = 0.001
interval = 50

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD([{'params':rgb_model.model.classifier.parameters()}, {'params':flow_model.model.classifier.parameters()}], lr=lr, momentum=0.9, weight_decay=0.0005 )
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005 )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
def train_model(model, n_epoch, optimizer, scheduler, train_loader, val_loader, model_dir):
    print('Start trianning')
    record = open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]), 'w+')
    for epoch in range(n_epoch):
        model.train()
        corrects_so_far = 0
        loss_so_far = 0
        count_so_far = 0
        loss = 0

        for idx, (rgb_buf, flow_buf, labels) in enumerate(train_loader):
            rgb_buf = rgb_buf.to(device)
            flow_buf = flow_buf.to(device)
            labels = labels.to(device)

            outputs = model(rgb_buf, flow_buf)
            loss = criterion(outputs, labels)

            _, pred_labels = torch.max(outputs, 1)
            print('pred labels ', pred_labels)
            print('true labels ', labels)

            loss_so_far += loss.item()
            corrects_so_far += torch.sum(pred_labels == labels).item()
            count_so_far += rgb_buf.size(0)

            if (idx+1) %  interval == 0:
                print('[acc-{:.4f}, loss-{:.4f} [{}/{}]'.format(corrects_so_far/count_so_far, loss_so_far/count_so_far, corrects_so_far, count_so_far))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('[train-{}/{}] [acc-{:.4f}, loss-{:.4f}] [{}/{}]\n'.format(epoch, n_epoch, corrects_so_far/count_so_far, loss_so_far/count_so_far, corrects_so_far, count_so_far))

        with open('./{}.txt'.format(os.path.basename(__file__).split('.')[0]),  'a+') as record:
            record.write('[train-{}/{}] [acc-{:.4f}, loss-{:.4f}] [{}/{}]\n'.
                    format(epoch, n_epoch, corrects_so_far/count_so_far, loss_so_far/count_so_far, corrects_so_far, count_so_far))

        model.eval()
        with torch.no_grad():
            corrects_so_far = 0
            count_so_far = 0
            loss_so_far = 0
            best_acc = 0

            for idx, (rgb_buf, flow_buf, labels) in enumerate(val_loader):
                rgb_buf = rgb_buf.to(device)
                flow_buf = flow_buf.to(device)
                labels = labels.to(device)

                outputs = model(rgb_buf, flow_buf)

                loss = criterion(outputs, labels)
                loss_so_far += loss.item()
                count_so_far += rgb_buf.size(0)
                _, pred_labels = torch.max(outputs, 1)
                corrects_so_far += torch.sum(pred_labels == labels).item()

            # may modify learning rate
            scheduler.step(loss)
            acc = corrects_so_far/count_so_far
            print('[val-{}/{}] [acc-{:.4f}, loss-{:.4f}] [{}/{}]\n'.format(
                epoch, n_epoch, acc, loss_so_far/count_so_far, corrects_so_far, count_so_far))

            with open(os.path.join(os.getcwd(), '{}.txt'.format(os.path.basename(__file__).split('.')[0])), 'a+') as record:
                record.write('[val-{}/{}] [acc-{:.4f}, loss-{:.4f}] [{}/{}]\n'.
                        format(epoch, n_epoch, corrects_so_far/count_so_far, loss_so_far/count_so_far, corrects_so_far, count_so_far))

            if corrects_so_far/count_so_far >= 0.84:
                if corrects_so_far/count_so_far > best_acc:
                    best_acc = corrects_so_far/count_so_far
                    try:
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)

                        param_dict = {'epoch':epoch,
                                'state_dict':model.state_dict(),
                                'best_acc':corrects_so_far/count_so_far,
                                'optimizer_param':optimizer.state_dict()}

                        torch.save(param_dict, os.path.join(model_dir,'two_stream_{:.4f}.pth'.format(corrects_so_far/count_so_far)))

                    except Exception as e:
                        print(str(e))
                        with open(os.path.join(os.getcwd(), '{}.txt'.format(os.path.basename(__file__).split('.')[0])), 'a+') as record:
                            record.write('[ERROR] ' + str(e) + '\n')

if __name__ == '__main__':
    model_dir = './trained_model'

    train_model(model,
                n_epoch,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                model_dir)
