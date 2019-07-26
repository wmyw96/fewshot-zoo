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
        print('Image Scale = {}'.format(np.max(self.inputs)))

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
            ind = np.argwhere(labels == i).reshape(-1)
            self.cb_index.append(ind)
            print('class {}: {} - {}'.format(i, np.min(ind), np.max(ind)))
            #print(self.labels[self.cb_index[i]])

        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.pointer = self.num_pairs
        #self.init_pointer()

    def len(self):
        return self.num_pairs

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]
        
            self.cb_index = []
            for i in range(self.nclass):
                ind = np.argwhere(self.labels == i).reshape(-1)
                self.cb_index.append(ind)
                #print('class {}: {} - {}'.format(i, np.min(ind), np.max(ind)))
            #print(self.labels[self.cb_index[i]])

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
        lb_cl = []
        for i in range(batch_class):
            cid = clslist[i]
            ind = np.random.permutation(self.cb_index[cid].shape[0])[:batch_size]
            dat = self.inputs[self.cb_index[cid][ind], :]
            #print(self.labels[self.cb_index[cid][ind]])
            dat = np.expand_dims(dat, 1)
            dat_cl.append(dat)
            lb_cl.append(self.labels[self.cb_index[cid][ind], :])
        return np.concatenate(dat_cl, axis=1), self.local_label(batch_class, batch_size), np.concatenate(lb_cl, 1)

    def get_support_query(self, batch_class, ns, nq, true_label=False):
        dat, label, _ = self.class_based_sample(batch_class, ns + nq)
        #print(label)
        #print(dat[:ns, :].shape, dat[ns:, :].shape)
        if true_label:
            return dat[:ns, :], dat[ns:, :], label[ns:, :], np.reshape(_, (-1, ))
        else:
            return dat[:ns, :], dat[ns:, :], label[ns:, :]

    def sample_from_class(self, clsid, batch_size):
        ind = np.random.permutation(self.cb_index[clsid].shape[0])[:batch_size]
        return self.inputs[self.cb_index[cid][ind], :]
    
