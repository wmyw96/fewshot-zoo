import numpy as np

def update_loss(fetch, loss):
    for key in fetch:
        if 'loss' in key:
            if key not in loss:
                loss[key] = []
            loss[key].append(fetch[key])

def print_log(title, epoch, loss):
    spacing = 10
    print_str = '{} epoch {}'.format(title, epoch)

    for k_, v_ in loss.items():
        if 'loss' in k_:
            value = np.around(np.mean(v_, axis=0), decimals=6)
            print_str += (k_ + ': ').rjust(spacing) + str(value) + ', '

    print_str = print_str[:-2]
    print(print_str)
