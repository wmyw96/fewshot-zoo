import numpy as np


class classfication_dataset(object):
    def __init__(self, inputs, labels, nclass, randomize=True, calc_weight=True):
        self.inputs = inputs
        self.labels = labels
        self.nclass = nclass
        if len(self.labels.shape) == 1:
            self.labels = np.reshape(self.labels,
                                     [self.labels.shape[0], 1])
        assert len(self.inputs) == len(self.labels)
        if calc_weight:
            self.weight = np.zeros((self.nclass), dtype=np.float32)
            for i in range(len(inputs)):
                self.weight[self.labels[i]] += 1.0
            self.weight /= np.sum(self.weight)
            assert np.abs(np.sum(self.weight) - 1) < 1e-9

        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def len(self):
        return self.num_pairs

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels.squeeze()

    def get_weight(self):
        return self.weight


class class_indexed_classification_dataset(object):
    def __init__(self):
        self.inputs = []
        self.nclass = 0
        self.weight = None

    def register(self):
        self.weight = np.zeros((self.nclass), dtype=np.float32)
        for i in range(len(inputs)):
            self.weight[i] = inputs[i].shape[0]
        self.weight /= np.sum(self.weight)
        assert np.abs(np.sum(self.weight) - 1) < 1e-9

    def add_class(self, inputs_cls):
        if self.weight is not None:
            print('Could not add class after registeration')
        self.inputs.append(inputs_cls)
        self.nclass += 1

    def next_batch(self, batch_class, batch_size):
        clslist = np.random.permutation(self.nclass)[:batch_class]
        for i in range(batch_class):
            return None
