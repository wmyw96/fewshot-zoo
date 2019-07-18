
def generate_params():
    nclass = 64

    data = {
        'dataset': 'mix-gaussian',
        'size': 10000, 
        'x_size': [64],
        'nclass': nclass,
        'radius': 2.0,
        'stddev': 0.5,
        'split': ['train'],
    }

    train = {
        'batch_size': 100,
        'num_epoches': 1000,
        'iter_per_epoch': 10000 // 100,
        'valid_interval': None,
    }

    lr = 1e-4
    reg_scale = 1e-8
    init = 'he'
    act = 'relu'
    z_dim = 2
    h_dim = 256

    network = {
        'nclass': nclass,
        'z_dim': z_dim,
        'use_decoder': True,
        'e_m_weight': 1.0,
        'lr': lr,
        'rec_weight': 1.0,
        'cls_weight': 1.0,
        'n_decay': 30,
        'weight_decay': 0.5
    }

    encoder = {
        'type': 'fc',
        'num_hidden': [h_dim]*2 + [z_dim],
        'activation': [act]*2 + [None],
        'init': [init]*3,
        'regularizer': [None]*3,
        'reg_scale': [reg_scale]*3
    }

    decoder = {
        'type': 'fc',
        'num_hidden': [h_dim]*2 + data['x_size'],
        'activation': [act]*2 + [None],
        'init': [init]*3,
        'regularizer': [None]*3,
        'reg_scale': [reg_scale]*3    
    }

    disc = {
        'lr': lr,
        'n_decay': 50,
        'weight_decay': 0.5,
        'gan-loss': 'wgan-gp',
        'type': 'fc',
        'gp_weight': 10.0,
        'n_critic': 5,
        'onehot_dim': 64,
        'nclass': nclass,
        'num_hidden': [h_dim*2]*3 + [1],
        'activation': [act]*3 + [None],
        'init': [init]*4,
        'regularizer': [None]*4,
        'reg_scale': [reg_scale]*4
    }

    embed = {
        'lr': 1e-1,
        'n_decay': 20,
        'weight_decay': 0.5,
        'type': 'gaussian',
        'stddev': 1.0
    }

    params = {
        'data': data,
        'train': train,
        'network': network,
        'encoder': encoder,
        'decoder': decoder,
        'disc': disc,
        'embedding': embed
    }

    return params
