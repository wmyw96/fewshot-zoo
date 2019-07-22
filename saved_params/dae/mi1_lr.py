
def generate_params():
    nclass = 64

    data = {
        'dataset': 'mini-imagenet',
        'data_dir': '../../data/mini-imagenet/',
        'split_dir': './splits/mini-imagenet',
        'x_size': [84, 84, 3],
        'nclass': nclass,
        'split': ['train', 'valid', 'test'],
    }

    train = {
        'batch_size': 600,
        'num_epoches': 500,
        'iter_per_epoch': 150,
        'valid_interval': 1,
    }

    test = {
        'n_way': [5, 5],
        'nq': 15,
        'shot': [5, 1],
        'num_episodes': 400,
    }

    lr = 1e-4
    reg_scale = 1e-8
    init = 'he'
    act = 'relu'
    h_dim = 512
    z_dim = 256
    #z_dim = 1600
    #h_dim = 256

    network = {
        'nclass': nclass,
        'z_dim': z_dim,
        'use_decoder': False,
        'e_m_weight': 0.1,
        'lr': 1e-3,
        'rec_weight': 0.0,
        'cls_weight': 1.0,
        'n_decay': 30,
        'weight_decay': 1.0
    }

    encoder = {
        'type': '4blockcnn',
        'num_hidden': [h_dim]*2 + [z_dim],
        'activation': [act]*2 + [None],
        'init': [init]*3,
        'regularizer': [None]*3,
        'reg_scale': [reg_scale]*3
    }

    '''decoder = {
        'type': 'fc',
        'num_hidden': [h_dim]*2 + data['x_size'],
        'activation': [act]*2 + [None],
        'init': [init]*3,
        'regularizer': [None]*3,
        'reg_scale': [reg_scale]*3    
    }'''

    disc = {
        'lr': lr,
        'n_decay': 50,
        'weight_decay': 1.0,
        'gan-loss': 'wgan-gp',
        'type': 'fc',
        'gp_weight': 10.0,
        'n_critic': 5,
        'onehot_dim': z_dim // 2,
        'nclass': nclass,
        'num_hidden': [1600]*3 + [1],
        'activation': [act]*3 + [None],
        'init': [init]*4,
        'regularizer': [None]*4,
        'reg_scale': [reg_scale]*4
    }

    embed = {
        'lr': 1.0,
        'n_decay': 20,
        'weight_decay': 0.5,
        'type': 'gaussian',
        'stddev': 1.0
    }

    params = {
        'data': data,
        'train': train,
        'test': test, 
        'network': network,
        'encoder': encoder,
        #'decoder': decoder,
        'disc': disc,
        'embedding': embed
    }

    return params
