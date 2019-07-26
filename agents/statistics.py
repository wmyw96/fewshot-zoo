import numpy as np
from scipy import stats


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
        'cnorm_mean': np.mean(centered_norm),
        'cnorm_std': np.std(centered_norm)
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

def guassian_test(x):
    '''
        x: [n, d]
    '''
    p_values = []
    for dim_id in range(x.shape[1]):
        w, p_value = stats.shapiro(x[:, dim_id])
        p_values.append(p_value)
    return {
        'pvalue_mean': np.mean(p_values),
        'pvalue_std': np.std(p_values)
    }

def correlation(x):
    '''
        x: [n, d]
    '''
    #x_ = x.transpose()
    x_ = (x - np.mean(x, 0, keepdims=True)) / np.std(x, 0, keepdims=True)
    print(np.std(x_, 0))
    print(x_)
    cor = np.cov(x_.transpose())
    print(cor)
    return cor    

def test_correlation():
#    inp = np.array([[11, 22, 33, 44, 55, 66, 77, 88, 99],
#        [10, 24, 30, 48, 50, 72, 70, 96, 90],
#        [91, 79, 72, 58, 53, 47, 34, 16, 10],
#        [99, 10, 98, 10, 17, 10, 77, 89, 10]])
#    inp = inp.transpose()
    inp = np.array([[0, 2], [1, 1], [2, 0]])
    output = np.array([[1, 0.98, -0.99, -0.17], [0.98, 1, -0.98, -0.18], [-0.99, -0.98, 1, 0.16], [-0.17, -0.18, 0.16, 1]])
    assert (np.max(np.abs(correlation(inp) - output)) < 0.02)

if __name__ == '__main__':
    test_correlation()
    
