#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import cv2 as cv
import pickle
import os
import argparse

def extract_rgb(video, save_dir, n_frame):
    # split image every 16 frames
    cap = cv.VideoCapture(video)
    print(cap.isOpened())
    folder = os.path.basename(video)

    all_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    img_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    buf = []
    # buf = np.zeros((n_frame, img_height, img_width, 3), dtype=np.float32)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    idx = 0
    suffix = 0
    try:
        while idx < all_frames:
            ret, rgb = cap.read()
            if ret == False:
                buf.clear()
                cap.release()
                break

            if idx % 4 == 0:
                buf.append(rgb)

                if len(buf) == n_frame:
                    os.mkdir(os.path.join(save_dir, folder+'_'+str(suffix)))
                    for num, rgb in enumerate(buf):
                        cv.imwrite(os.path.join(save_dir, folder+'_'+str(suffix), '{}_{:03d}.jpeg'.format(folder, num)), rgb)

                    suffix += 1
                    buf.clear()

                else:
                    continue

            idx += 1
    except Exception as e:
        print(e)
    finally:
        cap.release()

def process(video2path, split, save_root, n_frame, err_file):
    """
    video2path: pkl file to help find path
    split : train or val or test
    save_root : rgb file save postion, has 3 folder train  val test
    err_file : err record
    """
    if split == 'train':
        data_path = os.path.join(label_root, 'train_data')
    elif split == 'val':
        data_path = os.path.join(label_root, 'val_data')
    elif split == 'test':
        data_path = os.path.join(label_root, 'test_data')
    else:
        raise ValueError('no split')

    v2p = pickle.load(open(video2path, 'rb'))
    err_handle = open(err_file, 'w+')

    if not os.path.exists(os.path.join(save_root, split)):
        os.mkdir(os.path.join(save_root, split))

    # 
    try:
        for txt_file in os.listdir(data_path):
            # create folder based on class
            cls = txt_file.split('.')[0]
            if not os.path.exists(os.path.join(save_root, split, cls)):
                os.mkdir(os.path.join(save_root, split, cls))

            with open(os.path.join(data_path, txt_file), 'r') as txt_handle:
                for idx, video in enumerate(txt_handle.readlines()):

                    # video label need to split
                    video = video.strip().split(' ')[0]
                    print('split ', split, 'cls', txt_file.split('.')[0],'idx: ', idx)

                    if video not in v2p.keys():
                        err_handle.write(video +' not found ' + '\n')
                        raise RuntimeError('not find video ', video)

                    extract_rgb(v2p[video], os.path.join(save_root, split, cls), n_frame)

    except Exception as e:
        print(e)
        err_handle.write(str(e) + '\n')
    finally:
        err_handle.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_root', required=True)
    parser.add_argument('--video2path', required=True)
    parser.add_argument('--save_root', required=True)
    parser.add_argument('--n_frame', type=int, required=True)
    parser.add_argument('--err_file', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    video2path = args.video2path
    label_root = args.label_root
    save_root = args.save_root
    err_file = args.err_file
    n_frame = args.n_frame


    print('[*] processing train split now')
    process(video2path, 'train', save_root, n_frame, err_file)

    print('[*] processing val split now')
    process(video2path, 'val', save_root, n_frame, err_file)

    print('[*] processing test split now')
    process(video2path, 'test', save_root, n_frame, err_file)
