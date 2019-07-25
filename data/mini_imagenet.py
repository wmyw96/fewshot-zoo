import numpy as np
import cv2
import os
from tqdm import tqdm
from data.base import *
from skimage.transform import resize
from matplotlib.pyplot import imread
#im = imread(image.png)
#from scipy.misc import imresize
from scipy import ndimage


def read_and_resize(path):
    x = imread(path)
    x = resize(x, (84, 84))
    #print(x)
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return x
#read_and_resize = lambda x: cv2.resize(cv2.imread(x, 1), (84, 84))

def get_rotation(x):
    ret = []
    for angle in [0.0, 90.0, 180.0, 270.0]:
        ret.append(ndimage.rotate(x, angle, reshape=False, cval=0.0))
    return ret

def load_mini_imagenet(params):
    pd = params['data']
    data_dir = pd['data_dir']
    split_dir = pd['split_dir']
    print(split_dir)

    ret = {}

    splits = pd['split']
    for split in splits:
        gname = split
        if pd['rot']:
            gname += '-rot'
        try:
            #name = split
            #if split == 'valid':
            #    name = 'train'
            print('Find Cache File {}'.format(gname))            
            data = np.load(os.path.join(data_dir, gname + '-data.npy'))
            label = np.load(os.path.join(data_dir, gname + '-label.npy'))
            nclass = np.max(label) + 1
            print(data[0, :])
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
                im = read_and_resize(path)
                #print(im)
                if pd['rot']:
                    data += get_rotation(im)
                    label += [np.array([gid2clsid[gid]], dtype=np.int32)] * 4
                else:
                    data.append(im)
                    label.append(np.array([gid2clsid[gid]], dtype=np.int32))

            print('Split [{}]: {} classes, {} samples, Avg {} samples per class'.\
                format(split, cnt + 1, len(data), len(data) / (cnt + 1.0)))
            np.save(os.path.join(data_dir, gname + '-data.npy'), data)
            np.save(os.path.join(data_dir, gname + '-label.npy'), label)
            
            ret[split] = classfication_dataset(np.array(data), np.concatenate(label, 0), cnt + 1)
            
    return ret


def load_embed_mini_imagenet(params):
    pd = params['data']
    splits = pd['split']
    ret = {}
    for split in splits:
        gname = split
        data_path = pd['data_path'].format(split)
        label_path = pd['label_path'].format(split)
        x_dim = pd['x_size'][0]
        inputs = np.reshape(np.load(data_path), (-1, x_dim))
        labels = np.reshape(np.load(label_path), (-1, ))
        print('Split [{}]: number of classes = {}'.format(split, np.max(labels) + 1))
        ret[split] = classfication_dataset(inputs, labels, int(np.max(labels)) + 1)
    return ret

        
