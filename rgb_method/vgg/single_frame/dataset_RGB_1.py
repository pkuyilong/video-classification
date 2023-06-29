import os

import cv2 as cv
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

        self.resize_height = 280
        self.resize_width = 280
        self.crop_size = 224

        print('init video_list')
        self.video_list = [video for cls in os.listdir(os.path.join(self.root_dir, split)) for video in
                           os.listdir(os.path.join(self.root_dir, split, cls))]

        print('init video2path')
        self.video2path = {video: os.path.join(self.root_dir, split, cls, video) \
                           for cls in os.listdir(os.path.join(self.root_dir, split)) for video in
                           os.listdir(os.path.join(self.root_dir, split, cls))}

        print('init video2label')
        self.video2label = {video: label \
                            for label, cls in enumerate(os.listdir(os.path.join(self.root_dir, split))) for video in
                            os.listdir(os.path.join(self.root_dir, split, cls))}

        np.random.shuffle(self.video_list)

    def __getitem__(self, index):
        video = self.video_list[index]
        label = np.array(self.video2label[video])
        buf = self.load_frames(self.video2path[video], self.split)
        # return torch.from_numpy(buf), torch.from_numpy(label)
        return buf, label

    def __len__(self):
        return len(self.video_list)

    def load_frames(self, video_folder, split):
        frames = sorted([os.path.join(video_folder, img) for img in os.listdir(video_folder)])
        if split == 'train':
            idx = np.random.randint(5, 9)
            frame = cv.imread(frames[idx]).astype(np.float32)
            frame = cv.resize(frame, (self.crop_size, self.crop_size))
            frame = frame.transpose(2, 0, 1)
            frame[0] -= 123.68
            frame[1] -= 116.78
            frame[2] -= 103.94

            # tf = transforms.Compose([
            #     transforms.Resize((280, 280)),
            #     transforms.RandomCrop((224, 224)),
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #     ])

        elif split == 'val':
            idx = 5
            frame = cv.imread(frames[idx]).astype(np.float32)
            frame = cv.resize(frame, (self.crop_size, self.crop_size))
            frame = frame.transpose(2, 0, 1)
            frame[0] -= 123.68
            frame[1] -= 116.78
            frame[2] -= 103.94

            # tf = transforms.Compose([
            #     transforms.Resize((280, 280)),
            #     transforms.RandomCrop((224, 224)),
            #     transforms.ToTensor(),
            #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #     ])
            # frame = tf(frame)
        return frame


if __name__ == "__main__":
    train_data = VideoDataset(
        root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
        split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
        split='train',
        n_frame=16)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    print('train_lodaer', len(train_loader))

    for idx, (buf, label) in enumerate(train_loader):
        if idx == 1:
            break
        print('buf shape \n', buf.shape)
        print('buf \n ', buf)
        print(buf[buf > 0])
        print('label \n', label)
