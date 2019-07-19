import numpy as np
import cv2
import os
from tqdm import tqdm

read_and_resize = lambda x: cv2.resize(cv2.imread(x, 1), (84, 84))

def load_mini_imagenet(params):
    pd = params['data']
    data_dir = pd['data_dir']
    split_dir = pd['split_dir']
    print(split_dir)

    ret = {}

    splits = pd['split']
    for split in splits:
        csv_path = os.path.join(split_dir, split + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        gid2clsid = {}
        cnt = -1

        for lid in tqdm(range(len(lines)), desc='Split [{}]'.format(split)):
            l = lines[lid]
            name, gid = l.split(',')
            path = os.path.join(data_dir, name)
            if gid not in gid2clsid:
                cnt += 1
                gid2clsid[gid] = cnt
            data.append(np.expand_dims(read_and_resize(path), 0))
            label.append(gid2clsid[gid])

        print('Split [{}]: {} classes, {} samples, Avg {} samples per class'.
            format(cnt + 1, len(data), len(data) / (cnt + 1.0)))
        ret[splits] = classfication_dataset(np.concatenate(data, 0), np.concatenate(label, 0), cnt + 1)

    return ret