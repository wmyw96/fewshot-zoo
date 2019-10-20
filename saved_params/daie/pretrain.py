
def generate_params():
    nclass = 64

    data = {
        'rot': True,
        'dataset': 'mini-imagenet',
        'data_dir': '../../data/mini-imagenet/',
        'split_dir': './splits/mini-imagenet',
        'x_size': [84, 84, 3],
        'nclass': nclass,
        'split': ['train', 'valid', 'test'],
    }

    pretrain = {
        'type': '4blockcnn',
        'lr': 1e-3,
        'batch_size': 64 * 2,
        'num_epoches': 120,
        'iter_per_epoch': 400,
    }

    train = {
        'batch_size': 200,
        'num_epoches': 500,
        'iter_per_epoch': 360,
        'valid_interval': 1,
    }

    test = {
        'n_way': [5, 5],
        'nq': 15,
        'shot': [5, 1],
        'num_episodes': 600,
    }

    lr = 1e-4
    reg_scale = 1e-8
    init = 'xavier'
    act = 'relu'
    h_dim = 1600
    e_dim = 512
    z_dim = 512
    #z_dim = 1600
    #h_dim = 256

    network = {
        'update_pretrain': True,
        'nclass': nclass,
        'use_decoder': False,
        'e_m_weight': 1.0,
        'lr': lr,
        'rec_weight': 1.0,
        'z_dim': z_dim,
        'pretrain_weight': 1.0,
        'cls_weight': 1.0,
        'n_decay': 30,
        'weight_decay': 1.0,
        'metric': 'cos'
    }

    x2e = {
        'type': 'fc',
        'num_hidden': [h_dim]*1 + [e_dim],
        'activation': [act]*1 + [None],
        'init': [init]*2,
        'regularizer': [None]*2,
        'reg_scale': [reg_scale]*2,
        'dropout':[1.0,1.0]    
    }

    z2e = {
        'type': 'fc',
        'num_hidden': [h_dim]*1 + [e_dim],
        'activation': [act]*1 + [None],
        'init': [init]*2,
        'regularizer': [None]*2,
        'reg_scale': [reg_scale]*2,
        'dropout':[1.0,1.0]
    }

    e2z = {
        'type': 'fc',
        'num_hidden': [h_dim]*1 + [z_dim],
        'activation': [act]*1 + [None],
        'init': [init]*2,
        'regularizer': [None]*2,
        'reg_scale': [reg_scale]*2,
        'dropout':[1.0,1.0]
    }

    dlayer = 3
    disc = {
        'lr': lr,
        'n_decay': 50,
        'weight_decay': 1.0,
        'gan-loss': 'wgan-gp',
        'type': 'fc',
        'gp_weight': 10.0,
        'n_critic': 5,
        'onehot_dim': z_dim,
        'nclass': nclass,
        'num_hidden': [1600]*(dlayer-1) + [1],
        'activation': [act]*(dlayer-1) + [None],
        'init': [init]*dlayer,
        'regularizer': [None]*dlayer,
        'reg_scale': [reg_scale]*dlayer,
        'dropout': [1.0]*dlayer
    }

    z = {
        'lr': lr,
        'n_decay': 20,
        'weight_decay': 1.0,
        'type': 'gaussian',
        'stddev': 1.0
    }

    params = {
        'data': data,
        'train': train,
        'pretrain': pretrain,
        'test': test, 
        'network': network,
        'z2e': z2e,
        'x2e': x2e,
        'e2z': e2z,
        'disc': disc,
        'z': z
    }

    return params
