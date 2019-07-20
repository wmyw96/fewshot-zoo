from data.base import classfication_dataset
import numpy as np


def mixture_gausssian_dataset(n, x_dim, n_cluster, radius, stddev):
    embed = np.random.normal(0.0, radius, size=(n_cluster, x_dim))

    inputs, labels = np.zeros((n, x_dim), dtype=np.float32), np.zeros((n, ), dtype=np.int32)
    for i in range(n):
        # mixture of gaussian
        k = np.random.randint(0, n_cluster)
        k = int(k)

        labels[i] = k
        inputs[i] = np.random.normal(embed[k, :], stddev)

    return {'train': classfication_dataset(inputs, labels, n_cluster)}

