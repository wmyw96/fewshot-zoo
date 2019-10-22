
def generate_params():
    nclass = 64

    data = {
        'rot': False,
        'dataset': 'mini-imagenet',
        'data_dir': '../../data/mini-imagenet/',
        'split_dir': './splits/mini-imagenet',
        'data_path': '../../data/embed-mini-imagenet/{}_feature_norot.npy',
        'label_path': '../../data/embed-mini-imagenet/{}_label_norot.npy',
        'x_size': [84, 84, 3],
        'nclass': nclass,
        'split': ['train', 'valid', 'test'],
    }

    pretrain = {
        'type': 'resnet12',
        'lr': 1e-3,
        'batch_size': 64 * 2,
        'num_epoches': 120,
        'iter_per_epoch': 400,
        'test_iter': 400
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
    init = 'he'
    act = 'relu'
    h_dim = 1600
    z_dim = 512
    #z_dim = 1600
    #h_dim = 256

    network = {
        'nclass': nclass,
        'z_dim': z_dim,
        'use_decoder': False,
        'e_m_weight': 1.0,
        'lr': lr,
        'rec_weight': 0.0,
        'cls_weight': 1.0,
        'n_decay': 30,
        'weight_decay': 1.0,
        'metric': 'cos'
    }

    encoder = {
        'type': 'fc',
        'num_hidden': [h_dim]*1 + [z_dim],
        'activation': [act]*1 + [None],
        'init': [init]*2,
        'regularizer': [None]*2,
        'reg_scale': [reg_scale]*2,
        'dropout':[1.0,1.0]
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
        'onehot_dim': z_dim,
        'nclass': nclass,
        'num_hidden': [1600]*3 + [1],
        'activation': [act]*3 + [None],
        'init': [init]*4,
        'regularizer': [None]*4,
        'reg_scale': [reg_scale]*4,
        'dropout': [1.0]*4
    }

    embed = {
        'lr': lr,
        'n_decay': 20,
        'weight_decay': 1.0,
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
        'pretrain': pretrain,
        'disc': disc,
        'embedding': embed
    }

    return params
