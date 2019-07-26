import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def norm(embed):
    '''
       embed: [n, d]
    '''
    nanasa = np.mean(embed, 0, keepdims=True)
    norm = np.sqrt(np.sum(tf.square(embed), 1))
    centered_norm = np.sqrt(np.sum(np.square(embed - nanasa), 1))
    return {
        'norm_mean': np.mean(norm),
        'norm_std': np.std(norm),
        'norm_max': np.max(norm),
        'cnorm_mean': np.mean(centered_norm),
        'cnorm_std': np.std(centered_norm),
        'cnorm_max': np.max(centered_norm),
    }

def pairwise_distance(embed):
    '''
        embed: [n, d]
    '''
    x = np.expand_dims(embed, 1)
    y = np.expand_dims(embed, 0)
    dist = np.sqrt(tf.sum(np.square(x - y), 2))   # [n, n]
    return {
        'dist_mean': np.mean(dist),
        'dist_std': np.mean(dist)
    }

def gaussian_test(x):
    '''
        x: [n, d]
    '''
    p_values = []
    cnt_5 = 0
    for dim_id in range(x.shape[1]):
        w, p_value = stats.shapiro(x[:, dim_id])
        p_values.append(p_value)
        cnt_5 += p_value < 0.05
    return {
        'pvalue_mean': np.mean(p_values),
        'pvalue_std': np.std(p_values),
        'sign': (cnt_5 + 0.0) / x.shape[1]
    }


def correlation(x):
    '''
        x: [n, d]
    '''
    x_ = x.transpose()
    cor = np.cov(x_)
    norm = cor.diagonal()
    cor = cor / np.sqrt(np.expand_dims(norm, 0) * np.expand_dims(norm, 1))
    return {
        'cor_mean': np.mean(cor - cor.diagonal()),
    }


def test_correlation():
    inp = np.array([[11, 22, 33, 44, 55, 66, 77, 88, 99],
        [10, 24, 30, 48, 50, 72, 70, 96, 90],
        [91, 79, 72, 58, 53, 47, 34, 16, 10],
        [99, 10, 98, 10, 17, 10, 77, 89, 10]])
    inp = inp.transpose()
    #inp = np.array([[0, 2], [1, 1], [2, 0]])
    output = np.array([[1, 0.98, -0.99, -0.17], [0.98, 1, -0.98, -0.18], [-0.99, -0.98, 1, 0.16], [-0.17, -0.18, 0.16, 1]])
    assert (np.max(np.abs(correlation(inp) - output)) < 0.01)


def test_gaussian_test():
    a = np.random.normal(0, 1, size=(100, 1))
    p_1 = gaussian_test(a)
    print(p_1)
    assert p_1['pvalue_mean'] > 0.05
    b = np.random.normal(0, 3, size=(100, 1))
    pp = np.concatenate([a, b], 0)
    p_2 = gaussian_test(pp)
    print(p_2)
    assert p_2['pvalue_mean'] < 0.05


def tsne_visualization(x, y, path):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    x_tsne = tsne.fit_transform(X)
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(16, 16))
    for i in range(x_norm.shape[0]):
        plt.text(x_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                 fontdict={'weight': 'bold', 'size': 5})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path)

if __name__ == '__main__':
    test_correlation()
    test_gaussian_test()
