import numpy as np
import cv2
import os


read_and_resize = lambda x: cv2.resize(cv2.imread(x, 1), (width, height))

def load_mini_imagenet(params):
    pd = params['data']
    data_dir = pd['data_dir']
    split_dir = pd['split_dir']
    print(split_dir)

    ret = {}

    splits = pd['splits']
    for split in splits:
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        gid2clsid = {}
        cnt = -1

        for l in lines:
            name, gid = l.split(',')
            path = osp.join(data_dir, name)
            if gid not in gid2clsid:
                cnt += 1
                gid2clsid[gid] = cnt
            data.append(read_and_resize(path).expand_dims(0))
            label.append(gid2clsid[gid])

        print('Split [{}]: {} classes, {} samples, Avg {} samples per class'.
            format(cnt + 1, len(data), len(data) / (cnt + 1.0)))
        ret[splits] = classfication_dataset(np.concatenate(data, 0), np.concatenate(label, 0), cnt + 1)

    return ret