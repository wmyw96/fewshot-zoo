from data.base import *
from data.mix_normal import *


def load_dataset(params):
    if params['data']['dataset'] == 'mix-gaussian':
        return mixture_gausssian_dataset(params['data']['size'], 
                                         params['data']['x_size'][0], 
                                         params['data']['nclass'],
                                         params['data']['radius'], 
                                         params['data']['stddev'])
    else:
        raise ValueError('Dataset doesn\'t exist !!')
