import os
import cv2 as cv
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, root_dir, split_data, split):
        """
        root_dir : 存放数据的根目录
        split_data： 存放train val test的根目录
        split ：train  or val or test
        """
        super().__init__()
        self.root_dir = root_dir
        self.split_data = split_data
        self.split = split

        self.resize_height = 280
        self.resize_width = 280
        self.crop_size = 224


        print('init video_list')
        self.video_list = [video for cls in os.listdir(os.path.join(self.root_dir, self.split)) for video in os.listdir(os.path.join(self.root_dir, self.split, cls))]

        print('init video2path')
        self.video2path = {video : os.path.join(self.root_dir, self.split, cls, video) \
            for cls in os.listdir(os.path.join(self.root_dir, self.split)) for video in os.listdir(os.path.join(self.root_dir, self.split, cls)) }

        print('init video2label')
        self.video2label = {video : label \
            for label, cls in enumerate(os.listdir(os.path.join(self.root_dir, self.split))) for video in os.listdir(os.path.join(self.root_dir, self.split, cls)) }

        np.random.shuffle(self.video_list)

    def __getitem__(self, index):
        video = self.video_list[index]
        label = np.array(self.video2label[video])
        rgb_buf, flow_buf = self.load_frames(self.video2path[video])
        return torch.from_numpy(rgb_buf), torch.from_numpy(flow_buf), torch.from_numpy(label)

    def __len__(self):
        return len(self.video_list)

    def load_frames(self, video_folder):
        # return 1 rgb and 20 optical flow
        rgb_buf = None
        for name in os.listdir(video_folder):
            if name.startswith('rgb'):
                rgb_buf = cv.imread(os.path.join(video_folder,name)).astype(np.float32)
                rgb_buf = cv.resize(rgb_buf, (self.resize_height, self.resize_width))
                rgb_buf[..., 0] = rgb_buf[..., 0] - np.average(rgb_buf[..., 0])
                rgb_buf[..., 1] = rgb_buf[..., 1] - np.average(rgb_buf[..., 1])
                rgb_buf[..., 2] = rgb_buf[..., 2] - np.average(rgb_buf[..., 2])
                start_height = np.random.randint(0, rgb_buf.shape[0] - self.crop_size + 1)
                start_width = np.random.randint(0, rgb_buf.shape[1] - self.crop_size + 1)
                rgb_buf = rgb_buf[start_height : start_height+self.crop_size,
                        start_width : start_width+self.crop_size, :]
                rgb_buf = rgb_buf.transpose(2, 0, 1)

        flowx_files = sorted([os.path.join(video_folder, name) for name in os.listdir(video_folder) if name.startswith('flowx')])
        flowy_files = sorted([os.path.join(video_folder, name) for name in os.listdir(video_folder) if name.startswith('flowy')])

        cur_flowx_num = len(flowx_files)
        if cur_flowx_num == 0:
            print(video_folder, 'has no flowx frame')
        while cur_flowx_num < 10:
            print(video_folder, cur_flowx_num)
            need_to_fill = 10 - cur_flowx_num
            while need_to_fill > 0:
                flowx_files.append(flowx_files[-1])
                need_to_fill -= 1

        cur_flowy_num = len(flowy_files)
        if cur_flowy_num == 0:
            print(video_folder, 'has no flowy frame')
        while cur_flowy_num < 10:
            print(video_folder, cur_flowy_num)
            need_to_fill = 10 - cur_flowy_num
            while need_to_fill > 0:
                flowy_files.append(flowy_files[-1])
                need_to_fill -= 1

        flow_buf = np.empty((self.resize_height, self.resize_width, 20), np.dtype('float32'))

        for idx, (flowx, flowy) in enumerate(zip(flowx_files, flowy_files)):
            flow_x = cv.imread(flowx).astype(np.float32)
            flow_y = cv.imread(flowy).astype(np.float32)
            flow_x = cv.resize(flow_x, (self.resize_width, self.resize_height))
            flow_y = cv.resize(flow_y, (self.resize_width, self.resize_height))
            flow_x = np.max(flow_x, axis=2)
            flow_y = np.max(flow_y, axis=2)

            flow_buf[:, :, 2*idx] = flow_x
            flow_buf[:, :, 2*idx+1] = flow_y

        start_height = np.random.randint(0, flow_buf.shape[0] - self.crop_size + 1)
        start_width = np.random.randint(0, flow_buf.shape[1] - self.crop_size + 1)
        flow_buf = flow_buf[start_height : start_height+self.crop_size, start_width : start_width+self.crop_size, :]
        flow_buf = flow_buf.transpose(2, 0, 1)
        return (rgb_buf, flow_buf)

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv.flip(buffer[i], flipCode=1)
                buffer[i] = cv.flip(frame, flipCode=1)

        return buffer

if __name__ == "__main__":
    print('#'*80)
    # train_data = VideoDataset(
    #     root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb_flow_300',
    #     split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
    #     split='train')

    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
    # print('train_lodaer',len(train_loader))

    # for idx, (rgb_buf, flow_buf, label) in enumerate(train_loader):
    #     print('rgb_buf size is ', rgb_buf.size())
    #     print('flow_buf size is ', flow_buf.size())
    #     print('label is : ', label)
    #     if idx == 300:
    #         break

    val_data = VideoDataset(
        root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb_flow_300',
        split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
        split='val')

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=1)
    print('val_lodaer',len(val_loader))

    for idx, (rgb_buf, flow_buf, label) in enumerate(val_loader):
        print('rgb_buf size is ', rgb_buf.size())
        print('flow_buf size is ', flow_buf.size())
        print('label is : ', label)
