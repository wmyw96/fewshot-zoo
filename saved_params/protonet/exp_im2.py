
def generate_params():
    data = {
        'dataset': 'mini-imagenet',
        'data_dir': '../../data/mini-imagenet/',
        'split_dir': './splits/mini-imagenet',
        'x_size': [84, 84, 3],
        'nclass': 64,
        'rot': False,
        'split': ['train', 'valid', 'test'],
    }

    train = {
        'n_way': 30,
        'nq': 15,
        'shot': 1,
        'num_epoches': 200,
        'iter_per_epoch': 150,
        'valid_interval': 1,
    }

    test = {
        'n_way': 5,
        'nq': 15,
        'shot': 1,
        'num_episodes': 400,
    }

    lr = 1e-3
    z_dim = 64
    h_dim = 64

    network = {
        'split_feed': True,
        'model': 'protonet',
        'lr': lr,
        'h_dim': h_dim,
        'z_dim': z_dim,
        'n_decay': 20,
        'weight_decay': 0.5
    }

    params = {
        'data': data,
        'train': train,
        'test': test,
        'network': network
    }

    return params
