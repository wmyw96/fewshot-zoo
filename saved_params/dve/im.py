
def generate_params():
    nclass = 64

    data = {
        'rot': False,
        'dataset': 'mini-imagenet',
        'data_dir': '../../data/mini-imagenet/',
        'split_dir': './splits/mini-imagenet',
        'x_size': [84, 84, 3],
        'nclass': nclass,
        'split': ['train', 'valid', 'test'],
    }

    pretrain = {
        'lr': 1e-3,
        'batch_size': 64,
        'num_epoches': 120,
        'iter_per_epoch': 400,
    }

    batch_size = 128

    train = {
        'batch_size': batch_size,
        'num_epoches': 100,
        'iter_per_epoch': 600 * 64 // batch_size,
        'valid_interval': 1,
    }

    test = {
        'n_way': [5, 5],
        'nq': 15,
        'shot': [5, 1],
        'num_episodes': 600,
    }

    lr = 1e-2
    reg_scale = 1e-8
    init = 'xavier'
    act = 'relu'
    h_dim = 1600
    z_dim = 512
    alpha = 5e-3
    #z_dim = 1600
    #h_dim = 256
    n_decay = int(80 * (600.0 / batch_size))
    network = {
        'nclass': nclass,
        'fixed': True,
        'z_dim': z_dim,
        'use_decoder': True,
        'e_m_weight': alpha,
        'lr': lr,
        'rec_weight': alpha,
        'cls_weight': 1.0,
        'n_decay': n_decay,
        'decay_weight': 0.95,
        'metric': 'cos'
    }
    
    nlayer = 2

    encoder = {
        'type': 'fc',
        'num_hidden': [h_dim] * (nlayer - 1) + [z_dim * 2],
        'activation': [act] * (nlayer - 1)+ [None],
        'init': [init]*nlayer,
        'regularizer': [None]*nlayer,
        'reg_scale': [reg_scale]*nlayer,
        'dropout':[1.0]*2
    }

    decoder = {
        'type': 'fc',
        'num_hidden': [h_dim] * (nlayer - 1) + [z_dim],
        'activation': [act] * (nlayer - 1) + [None],
        'init': [init]*nlayer,
        'regularizer': [None]*nlayer,
        'reg_scale': [reg_scale]*nlayer,
        'dropout':[1.0]*2   
    }
   
    embed = {
        'lr': 1.0,
        'n_decay': n_decay,
        'decay_weight': 0.95,
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
