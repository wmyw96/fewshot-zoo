
def generate_params():
    data = {
        'dataset': 'mini-imagenet',
        'data_dir': '../../data/mini-imagenet/',
        'split_dir': './splits/mini-imagenet',
        'x_size': [84, 84, 3],
        'nclass': 64,
        'split': ['train', 'valid', 'test'],
    }

    train = {
        'n_way': 20,
        'nq': 15,
        'shot': 5,
        'num_epoches': 200,
        'iter_per_epoch': 96,
        'valid_interval': 1,
    }

    test = {
        'n_way': 5,
        'nq': 15,
        'shot': 5,
        'num_episodes': 200,
    }

    lr = 1e-3
    z_dim = 64
    h_dim = 64

    network = {
        'model': 'protonet',
        'lr': lr,
        'h_dim': h_dim,
        'z_dim': z_dim,
        'n_decay': 30,
        'weight_decay': 0.5
    }

    params = {
        'data': data,
        'train': train,
        'test': test,
        'network': network
    }

    return params
