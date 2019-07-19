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

        # calculate p(y) prior
        if calc_weight:
            self.weight = np.zeros((self.nclass), dtype=np.float32)
            for i in range(len(inputs)):
                self.weight[self.labels[i]] += 1.0
            self.weight /= np.sum(self.weight)
            assert np.abs(np.sum(self.weight) - 1) < 1e-9
        # class-based index
        self.cb_index = []
        for i in range(self.nclass):
            ind = np.argwhere(label == i).reshape(-1)
            self.cb_index.append(ind)

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

    def local_label(self, batch_class, batch_size):
        ret = np.zeros((batch_size, batch_class), dtype=np.int32)
        for _ in range(batch_size):
            for clsid in range(batch_class):
                ret[_, clsid] = clsid
        return ret

    def class_based_sample(self, batch_class, batch_size):
        clslist = np.random.permutation(self.nclass)[:batch_class]
        dat_cl = []
        for i in range(batch_class):
            cid = clslist[i]
            ind = np.random.permuation(self.cb_index[cid].shape[0])[:batch_size]
            dat = self.inputs[self.cb_index[cid][ind], :]
            dat = np.expand_dims(dat, 1)
            dat_cl.append(dat)
        return np.concatenate(dat_cl, axis=1), self.local_label(batch_class, batch_size)

    def get_support_query(self, batch_class, ns, nq):
        dat, label = self.class_based_sample(batch_class, ns + nq)
        return dat[:ns, :], dat[ns:, :], np.reshape(label, (-1))
