import numpy as np
import cv2
import os
from tqdm import tqdm
from data.base import *


read_and_resize = lambda x: cv2.resize(cv2.imread(x, 1), (84, 84))

def load_mini_imagenet(params):
    pd = params['data']
    data_dir = pd['data_dir']
    split_dir = pd['split_dir']
    print(split_dir)

    ret = {}

    splits = pd['split']
    for split in splits:
        try:
            name = split
            #if split == 'valid':
            #    name = 'train'
            
            data = np.load(os.path.join(data_dir, name + '-data.npy'))
            label = np.load(os.path.join(data_dir, name + '-label.npy'))
            nclass = np.max(label) + 1
            print('Load cached {}: {}, {}'.format(split, data.shape, label.shape))
            ret[split] = classfication_dataset(np.squeeze(data), np.squeeze(label), nclass)
        except:
            print('Could not find file, preprocessing...')
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
                #print(cnt)
                if gid not in gid2clsid:
                    cnt += 1
                    gid2clsid[gid] = cnt
                data.append(read_and_resize(path))
                label.append(np.array([gid2clsid[gid]], dtype=np.int32))

            print('Split [{}]: {} classes, {} samples, Avg {} samples per class'.\
                format(split, cnt + 1, len(data), len(data) / (cnt + 1.0)))
            np.save(os.path.join(data_dir, split + '-data.npy'), data)
            np.save(os.path.join(data_dir, split + '-label.npy'), label)
            
            ret[split] = classfication_dataset(np.concatenate(data, 0), np.concatenate(label, 0), cnt + 1)
            
    return ret
