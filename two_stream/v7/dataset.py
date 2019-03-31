import os
import cv2 as cv
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

class VideoDataset(Dataset):
    def __init__(self, dataset_path, split_data, split, multi_scale=True):
        """
        dataset_path : 存放数据的根目录
        split_data： 存放train val test的根目录
        split ：train  or val or test
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.split_data = split_data
        self.split = split
        self.multi_scale = True

        self.crop_size = 224

        print('init video_list')
        self.video_list = [video for cls in os.listdir(os.path.join(self.dataset_path, self.split))
                for video in os.listdir(os.path.join(self.dataset_path, self.split, cls))]

        print('init video2path')
        self.video2path = {video : os.path.join(self.dataset_path, self.split, cls, video)
            for cls in os.listdir(os.path.join(self.dataset_path, self.split))
            for video in os.listdir(os.path.join(self.dataset_path, self.split, cls)) }

        print('init video2label')
        self.video2label = {video : label \
            for label, cls in enumerate(sorted(os.listdir(os.path.join(self.dataset_path, self.split))))
            for video in os.listdir(os.path.join(self.dataset_path, self.split, cls)) }

        np.random.shuffle(self.video_list)

    def __getitem__(self, index):
        video = self.video_list[index]
        label = np.array(self.video2label[video])
        rgb_buf, flow_buf = self.load_frames(self.video2path[video])
        if self.split == 'train':
            if np.random.randn() > 0.5:
                self.horizon_flip(rgb_buf, flow_buf)

        # print('origin', rgb_buf.shape, flow_buf.shape)
        rgb_buf, flow_buf = self.random_crop(rgb_buf, flow_buf)
        # print('crop', rgb_buf.shape, flow_buf.shape)

        rgb_buf, flow_buf = self.to_tensor(rgb_buf, flow_buf)
        # print('to tensor', rgb_buf.shape, flow_buf.shape)
        return torch.from_numpy(rgb_buf), torch.from_numpy(flow_buf), torch.from_numpy(label)

    def __len__(self):
        return len(self.video_list)

    def load_frames(self, video_folder):
        # return 1 rgb and 32 optical flow
        start_height, start_width = 0, 0
        resize_width, resize_height = 0, 0
        flowx_files, flowy_files = list(), list()

        for name in os.listdir(video_folder):
            if name.startswith('rgb'):
                rgb_buf = cv.imread(os.path.join(video_folder,name)).astype(np.float32)
                width = rgb_buf.shape[1]
                height = rgb_buf.shape[0]

                if self.multi_scale:
                    resize_height = np.random.randint(360, height)
                    resize_width = int(resize_height / height * width) if int(resize_height / height * width) > 360 else 360
                else:
                    resize_height = 280
                    resize_width = 280

                rgb_buf = cv.resize(rgb_buf, (resize_width, resize_height))
                rgb_buf[..., 0] = rgb_buf[..., 0] - np.average(rgb_buf[..., 0])
                rgb_buf[..., 1] = rgb_buf[..., 1] - np.average(rgb_buf[..., 1])
                rgb_buf[..., 2] = rgb_buf[..., 2] - np.average(rgb_buf[..., 2])

        if rgb_buf is None:
            raise ValueError('not found any rgb images')

        start_idx = np.random.randint(0, 16)
        for file_name in os.listdir(video_folder):
            if file_name.startswith('flowx'):
                flowx_files.append(os.path.join(video_folder, file_name))
            if file_name.startswith('flowy'):
                flowy_files.append(os.path.join(video_folder, file_name))
        flowx_files = flowx_files[start_idx: start_idx + 16]
        flowy_files = flowy_files[start_idx: start_idx + 16]

        # [16, 224, 224]
        flow_buf = np.empty((resize_height, resize_width, 16), np.dtype('float32'))

        for idx, (flowx, flowy) in enumerate(zip(flowx_files, flowy_files)):
            flow_x = cv.imread(flowx, 0).astype(np.float32)
            flow_y = cv.imread(flowy, 0).astype(np.float32)
            # cv.resize( width, height)
            flow_x = cv.resize(flow_x, (resize_width, resize_height))
            flow_y = cv.resize(flow_y, (resize_width, resize_height))

            flow = np.max((flow_x, flow_y), axis=0)
            flow_buf[:, :, idx] = flow

        return (rgb_buf, flow_buf)

    def horizon_flip(self, rgb_buf, flow_buf):
        rgb_buf = cv.flip(rgb_buf, 1)
        for idx in range(flow_buf.shape[2]):
            flow_buf[:,:,idx] = cv.flip(flow_buf[:,:,idx], 1)
        return (rgb_buf, flow_buf)

    def random_crop(self, rgb_buf, flow_buf):
        # [224, 224, 3] [224, 224, 16]
        start_height = np.random.randint(0, rgb_buf.shape[0] - self.crop_size + 1)
        start_width = np.random.randint(0, rgb_buf.shape[1] - self.crop_size + 1)
        rgb_buf = rgb_buf[start_height:start_height+self.crop_size, start_width:start_width+self.crop_size, :]
        flow_buf = flow_buf[start_height:start_height+self.crop_size, start_width:start_width+self.crop_size, :]
        return (rgb_buf, flow_buf)

    def to_tensor(self, rgb_buf, flow_buf):
        rgb_buf = rgb_buf.transpose(2, 0, 1)
        flow_buf = flow_buf.transpose(2, 0, 1)
        flow_buf = flow_buf[np.newaxis, :, :, :]
        return rgb_buf, flow_buf


if __name__ == "__main__":
    print('#'*80)
    dataset_path = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/datasets/dataset4/data'
    split_data = '/home/datasets/mayilong/PycharmProjects/p55/two_stream/datasets/dataset4/split_data'

    train_data = VideoDataset(
        dataset_path=dataset_path,
        split_data=split_data,
        split='train',
        multi_scale=True
        )
    val_data = VideoDataset(
        dataset_path=dataset_path,
        split_data=split_data,
        split='val',
        multi_scale=False
        )
    test_data = VideoDataset(
        dataset_path=dataset_path,
        split_data=split_data,
        split='test',
        multi_scale=False
        )

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=4)


    for idx, (rgb_buf, flow_buf, label) in enumerate(train_loader):
        print('rgb_buf size is ', rgb_buf.size())
        print('flow_buf size is ', flow_buf.size())
        print('label is : ', label)
