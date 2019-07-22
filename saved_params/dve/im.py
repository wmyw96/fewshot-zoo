
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

    pretrain = {
        'lr': 1e-3,
        'batch_size': 400,
        'num_epoches': 50,
        'iter_per_epoch': 100,
    }

    train = {
        'batch_size': 600,
        'num_epoches': 500,
        'iter_per_epoch': 64,
        'valid_interval': 1,
    }

    test = {
        'n_way': [5, 5],
        'nq': 15,
        'shot': [5, 1],
        'num_episodes': 400,
    }

    lr = 1e-3
    reg_scale = 1e-8
    init = 'he'
    act = 'relu'
    h_dim = 1600
    z_dim = 1600
    #z_dim = 1600
    #h_dim = 256

    network = {
        'nclass': nclass,
        'z_dim': z_dim,
        'use_decoder': True,
        'e_m_weight': 0.01,
        'lr': lr,
        'rec_weight': 0.01,
        'cls_weight': 1.0,
        'n_decay': 40,
        'weight_decay': 0.5
    }

    encoder = {
        'type': 'fc',
        'num_hidden': [h_dim]*2 + [z_dim * 2],
        'activation': [act]*2 + [None],
        'init': [init]*3,
        'regularizer': [None]*3,
        'reg_scale': [reg_scale]*3
    }

    decoder = {
        'type': 'fc',
        'num_hidden': [h_dim]*2 + [z_dim],
        'activation': [act]*2 + [None],
        'init': [init]*3,
        'regularizer': [None]*3,
        'reg_scale': [reg_scale]*3    
    }

    embed = {
        'lr': 1.0,
        'n_decay': 40,
        'weight_decay': 0.2,
        'type': 'gaussian',
        'stddev': 1.0
    }

    params = {
        'data': data,
        'train': train,
        'test': test,
        'pretrain': pretrain, 
        'network': network,
        'encoder': encoder,
        'decoder': decoder,
        'embedding': embed
    }

    return params
