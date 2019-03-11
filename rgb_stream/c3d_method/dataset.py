import os
import cv2 as cv
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, root_dir, split_data, split, n_frame=16):
        """
        root_dir : 存放数据的根目录
        split_data： 存放train_data val_data test_data的根目录
        video2label: path to video2label.pkl
        split ：train  or val or test
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_frame = n_frame
        self.split_data = split_data

        self.resize_size = 168
        self.crop_size = 112

        print('init video_list')
        self.video_list = [video for cls in os.listdir(os.path.join(self.root_dir, split)) for video in os.listdir(os.path.join(self.root_dir, split, cls))]

        print('init video2path')
        self.video2path = {video : os.path.join(self.root_dir, split, cls, video) \
            for cls in os.listdir(os.path.join(self.root_dir, split)) for video in os.listdir(os.path.join(self.root_dir, split, cls)) }

        print('init video2label')
        self.video2label = {video : label \
            for label, cls in enumerate(os.listdir(os.path.join(self.root_dir, split))) for video in os.listdir(os.path.join(self.root_dir, split, cls)) }

        np.random.shuffle(self.video_list)

    def __getitem__(self, index):
        video = self.video_list[index]
        label = np.array(self.video2label[video])
        buf = self.load_frames(self.video2path[video])
        buf = self.crop(buf, self.crop_size)

        if self.split == 'train':
            buf = self.randomflip(buf)

        buf = self.normalize(buf)
        buf = buf.transpose((1, 0, 2, 3))
        return torch.from_numpy(buf), torch.from_numpy(label)

    def __len__(self):
        return len(self.video_list)


    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv.flip(buffer[i], flipCode=1)
                buffer[i] = cv.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0]], [[98.0]], [[102.0]]])
            buffer[i] = frame
        return buffer

    def load_frames(self, video_folder):
        frames = sorted([os.path.join(video_folder, img) for img in os.listdir(video_folder)])
        buf = np.empty((self.n_frame, 3, self.resize_size, self.resize_size), np.float32)

        for idx, frame in enumerate(frames):
            try:
                frame = cv.imread(frame)
                frame = cv.resize(frame, (self.resize_size, self.resize_size))
                frame = frame.transpose(2,0,1).astype(np.float32)
                buf[idx] = frame
            except Exception as e:
                print(e, video_folder)
        return buf

    def crop(self, buf, crop_size):
        height_index = np.random.randint(buf.shape[2] - crop_size)
        width_index = np.random.randint(buf.shape[3] - crop_size)
        buf = buf[:, :,  height_index:height_index + crop_size, width_index:width_index + crop_size]
        return buf


if __name__ == "__main__":
    train_data = VideoDataset(
        root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
        split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
        split='train',
        n_frame=16)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2)
    print('train_lodaer',len(train_loader))

    for idx, (buf, label) in enumerate(train_loader):
        if idx == 3:
            break
        print(buf.shape)
        print(label)
