import os
import numpy as np
from PIL import Image
from utils.store_utils import *
from setting import *

def label2array(classes, label):
    # convert label to array
    label_array = np.array([0 for i in range(len(classes))])
    label_array[label] = 1.0
    return label_array

def get_images_from_folder(image_folder):
    images = np.zeros((16, 224, 224, 3), dtype=np.uint8)
    n_frame = 0
    for item in os.listdir(image_folder):
        image = Image.open(os.path.join(image_folder, item))
        image = image.resize((224,224))
        image = np.array(image)
        images[n_frame, :, :, :] = image
        n_frame += 1
    images = images.transpose((3,0,1,2))
    return images


# def video2label(train_annotaion_txt):
#     video_label = {}
#     with open(train_annotaion_txt, 'r') as f:
#         all_lines = f.readlines()
#         # shuffle(all_lines)
#         for i, line in enumerate(all_lines):
#             line = line.strip().split(',')
#             video = line.pop(0)
#             label = list(map(lambda x:int(x), line))
#             if video not in video_label.keys():
#                 video_label[video] = label
#     return video_label



# 不允许调用， 要使用video2label 直接打开文件
def video2label(annotation_file):
    try:
        video_label = {}
        with open(annotation_file, 'r') as f:
            all_lines = f.readlines()

            for i, line in enumerate(all_lines):
                line = line.strip().split(',')
                video = line.pop(0)
                label = list(map(lambda x:int(x), line))
                if video not in video_label.keys():
                    video_label[video] = label

        store_pkl('../resource/train_val_video2label.pkl', video_label)

    except Exception as e:
        print(e)

# 不允许调用， 要使用video2label 直接打开文件
def video2path(train_val_video_path):
    path = {}
    for d in os.listdir(train_val_video_path):
        for video_name in os.listdir(os.path.join(train_val_video_path, d)):
            path.update({video_name :
                os.path.join(train_val_video_path, d, video_name)})

    store_pkl('../resource/train_val_video2path.pkl', path)


if __name__ == '__main__':
    print("*"*80)

    pass
