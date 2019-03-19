#! /bin/sh
#
# train.sh
# Copyright (C) 2019 mayilong <mayilong@img>
#
# Distributed under terms of the MIT license.
#


python train.py \
--dataset_path='/home/datasets/mayilong/PycharmProjects/p55/two_stream/datasets/dataset3/data'   \
--split_data='/home/datasets/mayilong/PycharmProjects/p55/two_stream/dadatasets/dataset3/split_data' \
--batch_size=16 \
--lr=0.0005 \
--n_epoch=1000 \
--model_dir='./trained_model' \

