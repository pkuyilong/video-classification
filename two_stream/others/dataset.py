import os
import cv2 as cv
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def __init__(self, root_dir, split_data, split, ty='rgb', n_frame=1):
        """
        root_dir : 存放数据的根目录
        split_data： 存放train_data val_data test_data的根目录
        video2label: path to video2label.pkl
        split ：train  or val or test
        """
        super().__init__()
        self.root_dir = root_dir
        self.split_data = split_data
        self.split = split
        self.ty = ty
        self.n_frame = n_frame

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 280
        self.resize_width = 280
        self.crop_size = 224


        print('init video_list')
        self.video_list = [video for cls in os.listdir(os.path.join(self.root_dir, self.split))
                for video in os.listdir(os.path.join(self.root_dir, self.split, cls))]

        print('init video2path')
        self.video2path = {video : os.path.join(self.root_dir, self.split, cls, video)
            for cls in os.listdir(os.path.join(self.root_dir, self.split))
            for video in os.listdir(os.path.join(self.root_dir, self.split, cls)) }

        print('init video2label')
        self.video2label = {video : label \
            for label, cls in enumerate(os.listdir(os.path.join(self.root_dir, split)))
            for video in os.listdir(os.path.join(self.root_dir, self.split, cls)) }

        np.random.shuffle(self.video_list)

    def __getitem__(self, index):
        video = self.video_list[index]
        label = np.array(self.video2label[video])
        buf = self.load_frames(self.video2path[video], self.ty, self.n_frame)
        return torch.from_numpy(buf), torch.from_numpy(label)

    def __len__(self):
        return len(self.video_list)

    def load_frames(self, video_folder, ty='rgb', n_frame=1):
        if ty == 'rgb':
            buf = None
            for name in os.listdir(video_folder):
                if name.startswith('rgb'):
                    buf = cv.imread(os.path.join(video_folder,name)).astype(np.float32)
                    buf = cv.resize(buf, (self.resize_height, self.resize_width))
                    buf[..., 0] = buf[..., 0] - np.average(buf[..., 0])
                    buf[..., 1] = buf[..., 1] - np.average(buf[..., 1]
                    buf[..., 2] = buf[..., 2] - np.average(buf[..., 2]

                    start_height = np.random.randint(0, buf.shape[0] - self.crop_size + 1)
                    start_width = np.random.randint(0, buf.shape[1] - self.crop_size + 1)
                    buf = buf[start_height : start_height+self.crop_size,
                            start_width : start_width+self.crop_size, :]
                    buf = buf.transpose(2, 0, 1)
            return buf

        elif ty == 'flow':
            flowx_files = sorted([os.path.join(video_folder, name) for name in os.listdir(video_folder) if name.startswith('flowx')])
            flowy_files = sorted([os.path.join(video_folder, name) for name in os.listdir(video_folder) if name.startswith('flowy')])

            buf = np.empty((self.resize_height, self.resize_width, n_frame), np.dtype('float32'))
            for idx, (flowx, flowy) in enumerate(zip(flowx_files, flowy_files)):
                flow_x = cv.imread(flowx).astype(np.float32)
                flow_y = cv.imread(flowy).astype(np.float32)
                flow_x = cv.resize(flow_x, (self.resize_width, self.resize_height))
                flow_y = cv.resize(flow_y, (self.resize_width, self.resize_height))
                flow_x = np.max(flow_x, axis=2)
                flow_y = np.max(flow_y, axis=2)

                buf[:, :, 2*idx] = flow_x
                buf[:, :, 2*idx+1] = flow_y

            start_height = np.random.randint(0, buf.shape[0] - self.crop_size + 1)
            start_width = np.random.randint(0, buf.shape[1] - self.crop_size + 1)
            buf = buf[start_height : start_height+self.crop_size, start_width : start_width+self.crop_size, :]
            buf = buf.transpose(2, 0, 1)
            return buf
        else:
            raise ValueError('ty is not in rgb or flow')

    def randomflip(self, buffer):
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

    # def _crop(self, buffer, n_frame, crop_size):
    #     height_index = np.random.randint(buffer.shape[1] - crop_size)
    #     width_index = np.random.randint(buffer.shape[2] - crop_size)

    #     buffer = buffer[:, height_index:height_index + crop_size, width_index:width_index + crop_size, :]
    #     return buffer

if __name__ == "__main__":
    print('#'*80)
    train_data = VideoDataset(
        root_dir='/home/datasets/mayilong/PycharmProjects/p55/data/rgb_flow_300',
        split_data='/home/datasets/mayilong/PycharmProjects/p55/data/split_data',
        split='train',
        ty='flow',
        n_frame=20)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    print('train_lodaer',len(train_loader))

    for idx, (buf, label) in enumerate(train_loader):
        print('buf size is ', buf.size())
        print('label is : ', label)

        # if idx == 3:
        #     break
