import numpy as np
import os
from data.base import *


def update_loss(fetch, loss, need_loss=True):
    for key in fetch:
        if ('loss' in key) or (not need_loss):
            #if not need_loss:
            #    print(key)
            if key not in loss:
                loss[key] = []
            #print(fetch[key])
            loss[key].append(fetch[key])
    #print(fetch)
    #print(loss)

def print_log(title, epoch, loss):
    spacing = 10
    print_str = '{} epoch {}   '.format(title, epoch)

    for i, (k_, v_) in enumerate(loss.items()):
        if 'loss' in k_:
            #print('key = {}'.format(k_))
            value = np.around(np.mean(v_, axis=0), decimals=6)
            print_str += (k_ + ': ').rjust(spacing) + str(value) + ', '

    print_str = print_str[:-2]
    print(print_str)


def split_dataset(data, ratio):
    inputs, labels, nclass = data.inputs, data.labels, data.nclass
    ndata = inputs.shape[0]
    ind = np.random.permutation(ndata)
    inputs = inputs[ind, :]
    labels = labels[ind, :]
    melody = int(ndata * ratio)
    train = classfication_dataset(inputs[:melody, :], labels[:melody], nclass)
    test = classfication_dataset(inputs[melody:, :], labels[melody:], nclass)
    return train, test


class LogWriter(object):
    def __init__(self, dir, name):
        self.dir = dir
        if os.path.exists(self.dir):
            pass
        else:
            os.makedirs(self.dir)
        self.file_path = os.path.join(dir, name)
        # Clean the log file
        f = open(self.file_path, 'w')
        f.truncate()
        f.close()

    def print(self, epoch, domain, loss):
        spacing = 20
        print_str = 'Epoch {}   ({})\n'.format(epoch, domain)

        for i, (k_, v_) in enumerate(loss.items()):
            if True:
                value = np.around(np.mean(v_, axis=0), decimals=6)
                print_str += (k_ + ': ').rjust(spacing) + str(value) + '\n'
        print_str += '\n'
        with open(self.file_path, 'a') as f:
            f.write(print_str)
