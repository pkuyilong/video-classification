import os
import cv2 as cv
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.store_utils import parse_pkl


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

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 168
        self.resize_width = 168
        self.crop_size = 112

        # self.video2label = parse_pkl(os.path.join(video2label_root, self.split+'_video2label.pkl'))

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
        buf = self.crop(buf, self.n_frame, self.crop_size)

        if self.split == 'test':
            # Perform data augmentation
            buf = self.randomflip(buf)
        buf = self.normalize(buf)
        buf = self.to_tensor(buf)
        return torch.from_numpy(buf), torch.from_numpy(label)

    def __len__(self):
        return len(self.video_list)

    def get_split(self):
        root_dir = self.root_dir
        # split_data = self.split_data
        # split = self.split

        # unsplit_video_set = set()
        # data_path = os.path.join(split_data, split+'_data')
        # for txt_file in os.listdir(data_path):
        #     handle = open(os.path.join(data_path, txt_file), 'r')
        #     for idx, video  in enumerate(handle.readlines()):
        #         video = video.strip()
        #         unsplit_video_set.add(video)
        # for video in os.listdir(root_dir):
        #     name = video.strip().split('_')[0].strip()
        #     if name in unsplit_video_set:
        #         video_list.append(video)

        video_list = [video for video in os.listdir(root_dir)]
        return video_list

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv.flip(buffer[i], flipCode=1)
                buffer[i] = cv.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, video_folder):
        frames = sorted([os.path.join(video_folder, img) for img in os.listdir(video_folder)])
        buf = np.empty((self.n_frame, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        i = 0
        for idx, frame in enumerate(frames):
            try:
                img = cv.imread(frame)
                frame = np.array(cv.resize(img, (self.resize_width, self.resize_height))).astype(np.float32)
                buf[i] = frame
                i += 1
            except Exception as e:
                print(e, video_folder)
        return buf

    def crop(self, buffer, n_frame, crop_size):
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[:, height_index:height_index + crop_size, width_index:width_index + crop_size, :]
        return buffer


if __name__ == "__main__":
    train_data = VideoDataset(
        root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb',
        split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
        split='train',
        n_frame=16)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1)
    print('train_lodaer',len(train_loader))

    for idx, (buf, label) in enumerate(train_loader):
        if idx == 30:
            break
        print(buf.shape)
        print(label)
