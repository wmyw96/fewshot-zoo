
def generate_params():
    nclass = 64

    data = {
        'dataset': 'embed-mini-imagenet',
        'data_dir': '../../data/mini-imagenet/',
        'split_dir': './splits/mini-imagenet',
        'data_path': '../../data/embed-mini-imagenet/{}_feature_norot.npy',
        'label_path': '../../data/embed-mini-imagenet/{}_label_norot.npy',
        'x_size': [1600],
        'nclass': nclass,
        'split': ['train', 'valid', 'test'],
    }

    train = {
        'batch_size': 128,
        'num_epoches': 100,
        'iter_per_epoch': 520,
        'valid_interval': 2,
    }

    test = {
        'n_way': [5, 5],
        'nq': 15,
        'shot': [5, 1],
        'num_episodes': 600,
    }

    lr = 2e-4
    reg_scale = 1e-8
    init = 'he'
    act = 'leaky_relu'
    disc_init = 'he'
    disc_act = 'leaky_relu'
    h_dim = 1600
    z_dim = 512
    #z_dim = 1600
    #h_dim = 256

    network = {
        'nclass': nclass,
        'z_dim': z_dim,
        'use_decoder': True,
        'e_m_weight': 6.0,
        'lr': lr,
        'rec_weight': 0.0,
        'cls_weight': 1.0,
        'n_decay': 30,
        'weight_decay': 1.0,
        'metric': 'l2'
    }

    encoder = {
        'type': 'fc',
        'num_hidden': [h_dim]*1 + [z_dim],
        'activation': [act]*1 + [None],
        'init': [init]*2,
        'regularizer': [None]*2,
        'reg_scale': [reg_scale]*2
    }

    decoder = {
        'type': 'fc',
        'num_hidden': [h_dim]*1 + [1600],
        'activation': [act] * 1 + [None],
        'init': [init] * 2,
        'regularizer': [None]*2,
        'reg_scale': [reg_scale]*2
    }

    disc = {
        'em_weight': 6.0,
        'lr': lr,
        'n_decay': 50,
        'weight_decay': 1.0,
        'gan-loss': 'gan',
        'type': 'fc',
        'n_critic': 1,
        'onehot_dim': z_dim,
        'nclass': nclass,
        'num_hidden': [1600]*3 + [1],
        'activation': [disc_act]*3 + [None],
        'init': [disc_init]*4,
        'regularizer': [None]*4,
        'reg_scale': [reg_scale]*4
    }

    embed = {
        'lr': lr,
        'n_decay': 20,
        'weight_decay': 1.0,
        'type': 'rgaussian',
        'stddev': 0.5
    }

    params = {
        'data': data,
        'train': train,
        'test': test, 
        'network': network,
        'encoder': encoder,
        'decoder': decoder,
        'disc': disc,
        'embedding': embed
    }

    return params
