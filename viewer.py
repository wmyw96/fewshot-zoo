import re
import numpy as np
import matplotlib.pyplot as plt
from utils import color_set as clrset


class LogViewer(object):
    def __init__(self):
        self.data = {}

    # domain is a table
    # item: subdomain/itemname
    # recommend not to use '(', ')', '/' in item/domain name

    def load_log(self, domain, file_path, empty=None):
        self.data[domain] = {}

        content = []
        hier_name2idx = {}
        subdomain_name2idx = {}
        num_subdomain = 0
        item_name2idx = {}
        num_item = 0

        epoch_pattern = re.compile(r'(?<=Epoch )\d+\.?\d*')
        subdomain_pattern = re.compile(r'\((.+?)\)')
        with open(file_path, 'r') as f:
            subdomain = ''

            epoch = -1
            for line in f.readlines():
                if line[:5] == 'Epoch':
                    num_list = epoch_pattern.findall(line)
                    assert len(num_list) == 1
                    cur_epoch = int(num_list[0])

                    subdomain_list = subdomain_pattern.findall(line)
                    subdomain_name = subdomain_list[0]

                    epoch = cur_epoch
                    subdomain = subdomain_name
                    if subdomain not in subdomain_name2idx:
                        subdomain_name2idx[subdomain] = num_subdomain
                        num_subdomain += 1
                        hier_name2idx[subdomain] = {}
                elif len(line) > 15:
                    item_name, value = line.split(':')
                    item_name = item_name.strip(' ')
                    value = float(value)

                    item_full_name = subdomain + '/' + item_name
                    #rint(item_full_name)

                    if item_name not in hier_name2idx[subdomain]:
                        hier_name2idx[subdomain][item_name] = num_item
                        item_name2idx[item_full_name] = num_item
                        num_item += 1
                        content.append([])

                    item_idx = item_name2idx[item_full_name]

                    # fill with empty
                    while len(content[item_idx]) < epoch:
                        content[item_idx].append(content[item_idx][len(content[item_idx]) - 1])
                    content[item_idx].append(value)

        # late padding
        epoch += 1
        for i in range(num_item):
            while len(content[i]) < epoch:
                content[i].append(content[i][len(content[i]) - 1])
            content[i] = np.expand_dims(np.array(content[i]), 0)

        print('[Domain] ({})'.format(domain))
        for key in hier_name2idx.keys():
            print_str = '- [Subdomain] ({}): '.format(key)
            for item in hier_name2idx[key].keys():
                print_str += item + ' '
            print(print_str)


        self.data[domain] = {
            'content': np.concatenate(content, 0),
            'hier_name2id': hier_name2idx,
            'item_name2id': item_name2idx
        }
        print(self.data[domain]['content'].shape)

    def get_log(self, domain, subdomain, item):
        idx = self.data[domain]['hier_name2id'][subdomain][item]
        print('Find Item Index = {} in Domain {}'.format(idx, domain))
        print(self.data[domain]['content'].shape)
        return self.data[domain]['content'][idx, :]

    def plot(self, menu, ax=None):
        if ax is None:
            ax = plt
        cnt = 0
        for item in menu:
            domain, subdomain, item = item
            if item in self.data[domain]['hier_name2id'][subdomain]:
                log = self.get_log(domain, subdomain, item)
                horizon = log.shape[0]
                plt.plot(np.arange(horizon), log, linewidth=0.5, marker='s', ms=2.0, color=clrset[cnt], 
                         label=domain + '/' + subdomain + '/' + item)
            else:
                log_mean = self.get_log(domain, subdomain, item + '_mean')
                horizon = log_mean.shape[0]
                plt.plot(np.arange(horizon), log_mean, linewidth=0.5, marker='s', ms=2.0, color=clrset[cnt], 
                         label=domain + '/' + subdomain + '/' + item)
                log_std = self.get_log(domain, subdomain, item + '_std')
                log_up = log_mean + 3 * log_std
                log_low = log_mean - 3 * log_std
                plt.fill_between(np.arange(horizon), log_low, log_up, color=clrset[cnt], alpha = 0.4)

            cnt += 1
        ax.legend(fontsize=6)
        ax.xlabel('epoch')
        ax.ylabel('value')
        #plt.show()

    def self_plot(self, menu):
        cnt = 0
        for item in menu:
            name, val = item
            horizon = val.shape[0]
            plt.plot(np.arange(horizon), val, marker='s-', ms=2.0, color=clrset[cnt], label=name)
            cnt += 1
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.show()        


a = LogViewer()
a.load_log('dae', 'logs/dae.mi1_1ec08_02_10_26/main.log')
a.load_log('dae-lr1', 'logs/dae.mi1_1ec_elr107_29_13_06/main.log')
a.load_log('dve', 'logs/dve.im07_29_13_06/main.log')
a.load_log('proto', 'logs/protonet.exp_im207_29_16_36/main.log')
a.load_log('vmf', 'logs/dae.mi1_1vmf08_02_11_59/main.log')


####### DAE


plot = 'dae'
b = LogViewer()
### fixed & well init mu
# b.load_log('dae', 'logs/dae.mi1_1ec_nocls_rg08_05_10_13/main.log')

### fixed & guassian init mu


### best
#b.load_log('dae', 'logs/dae.mi1_1ec08_04_16_20/main.log')

### overall
#b.load_log('dae', 'logs/dae.mi1_1ec_lr-408_05_15_49/main.log') 

### cls_weight=10
#b.load_log('dae', 'logs/dae.mi1_1ec_lr-4_c1008_05_16_21/main.log')

### s=0.3
b.load_log('dae', 'logs/dae.mi1_1ec_lr-4_sz308_05_16_36/main.log')

### adaptive s
#b.load_log('dae', 'logs/dae.mi1_1ec_lr-4_ads08_05_17_56/main.log')

result = [
    ('dae', 'train-stat', 'embed_norm'),
    ('dae', 'train-stat', 'embed_cnorm'),
    ('dae', 'train-stat', 'est_norm'),
    ('dae', 'train-stat', 'est_cnorm'),
]
result_db = [
    ('dae', 'train-stat', 'davies_bouldin'),
    ('dae', 'val-stat', 'davies_bouldin'),
    ('dae', 'test-stat', 'davies_bouldin'),
]
result_gaus = [
    ('dae', 'train-stat', 'sign'),
    ('dae', 'val-stat', 'sign'),
    ('dae', 'test-stat', 'sign')
]
result_gaus2 = [
    ('dae', 'train-stat', 'mean_div'),
]
result_gaus3 = [
    ('dae', 'train-stat', 'mean_std'),
    ('dae', 'val-stat', 'mean_std'),
    ('dae', 'test-stat', 'mean_std'),    
]
result_acc = [
    ('dae', 'training', 'acc_loss'),
    ('dae', 'val', '5-way 1-shot val'),
    ('dae', 'val', '5-way 1-shot test'),
    ('dae', 'val', '5-way 5-shot val'),
    ('dae', 'val', '5-way 5-shot test'),
]
result_inclass = [
    ('dae', 'train-stat', 'inclass_cnorm'),
    ('dae', 'val-stat', 'inclass_cnorm'),
]
result_guascor = [
    ('dae', 'train-stat', 'cor_mean'),
    ('dae', 'val-stat', 'cor_mean'),
    ('dae', 'test-stat', 'cor_mean'),
]
result_exclass = [
    ('dae', 'train-stat', 'est_dist'),
    ('dae', 'val-stat', 'est_dist'),
    ('dae', 'test-stat', 'est_dist')
]

if plot == 'dae':
    plt.figure()
    plt.subplot(3,3,1)
    b.plot(result_db)
    plt.subplot(3,3,4)
    b.plot(result_acc)
    plt.subplot(3,3,2)
    b.plot(result_gaus)
    plt.subplot(3,3,3)
    b.plot(result_gaus2)
    plt.subplot(3,3,5)
    b.plot(result_gaus3)
    plt.subplot(3,3,6)
    b.plot(result)
    plt.subplot(3,3,7)
    b.plot(result_inclass)
    plt.subplot(3,3,8)
    b.plot(result_guascor)
    plt.subplot(3,3,9)
    b.plot(result_exclass)
    plt.show()



############# DVE
b.load_log('dve', 'logs/dve.im_ec08_04_16_14/main.log')
result = [
    ('dve', 'train-stat', 'embed_norm'),
    ('dve', 'train-stat', 'embed_cnorm'),
    ('dve', 'train-stat', 'est_norm'),
    ('dve', 'train-stat', 'est_cnorm'),
]
result_db = [
    ('dve', 'train-stat', 'davies_bouldin'),
    ('dve', 'val-stat', 'davies_bouldin'),
    ('dve', 'test-stat', 'davies_bouldin'),
]
result_gaus = [
    ('dve', 'train-stat', 'sign'),
    ('dve', 'val-stat', 'sign'),
    ('dve', 'test-stat', 'sign'),    
    ('dve', 'train-stat', 'truezsign'),
    ('dve', 'val-stat', 'truezsign'),
    ('dve', 'test-stat', 'truezsign'),
]
result_gaus2 = [
    ('dve', 'train-stat', 'mean_div'),
]
result_gaus3 = [
    ('dve', 'train-stat', 'mean_std'),
    ('dve', 'train-stat', 'truez_mean_std'),
    ('dve', 'val-stat', 'mean_std'),
    ('dve', 'val-stat', 'truez_mean_std'),
    ('dve', 'test-stat', 'mean_std'),    
    ('dve', 'test-stat', 'truez_mean_std'),
]
result_acc = [
    ('dve', 'training', 'acc_loss'),
    ('dve', 'val', '5-way 1-shot val'),
    ('dve', 'val', '5-way 1-shot test'),
    ('dve', 'val', '5-way 5-shot val'),
    ('dve', 'val', '5-way 5-shot test'),
]
result_inclass = [
    ('dve', 'train-stat', 'inclass_cnorm'),
    ('dve', 'val-stat', 'inclass_cnorm'),
]

result_guascor = [
    ('dve', 'train-stat', 'cor_mean'),
    ('dve', 'val-stat', 'cor_mean'),
    ('dve', 'test-stat', 'cor_mean'),
]
result_exclass = [
    ('dve', 'train-stat', 'est_dist'),
    ('dve', 'val-stat', 'est_dist'),
    ('dve', 'test-stat', 'est_dist')
]


if plot == 'dve':
    plt.figure()
    plt.subplot(3,3,1)
    b.plot(result_db)
    plt.subplot(3,3,4)
    b.plot(result_acc)
    plt.subplot(3,3,2)
    b.plot(result_gaus)
    plt.subplot(3,3,3)
    b.plot(result_gaus2)
    plt.subplot(3,3,5)
    b.plot(result_gaus3)
    plt.subplot(3,3,6)
    b.plot(result)
    plt.subplot(3,3,7)
    b.plot(result_inclass)
    plt.subplot(3,3,8)
    b.plot(result_guascor)
    plt.subplot(3,3,9)
    b.plot(result_exclass)
    plt.show()



############# ProtoNet
b.load_log('proto', 'logs/protonet.exp_im208_04_17_09/main.log')
result = [
    ('proto', 'train-stat', 'est_norm'),
    ('proto', 'train-stat', 'est_cnorm'),
]
result_db = [
    ('proto', 'train-stat', 'davies_bouldin'),
    ('proto', 'val-stat', 'davies_bouldin'),
    ('proto', 'test-stat', 'davies_bouldin'),
]
result_gaus = [
    ('proto', 'train-stat', 'sign'),
    ('proto', 'val-stat', 'sign'),
    ('proto', 'test-stat', 'sign')
]
result_gaus3 = [
    ('proto', 'train-stat', 'mean_std'),
    ('proto', 'val-stat', 'mean_std'),
    #('dve', 'val-stat', 'truez_mean_std'),
    ('proto', 'test-stat', 'mean_std'),    
    #('dve', 'test-stat', 'truez_mean_std'),
]
result_acc = [
    ('proto', 'training', 'acc_loss'),
    ('proto', 'valid', 'valid_acc'),
    ('proto', 'valid', 'test_acc'),
]
result_inclass = [
    ('proto', 'train-stat', 'inclass_cnorm'),
    ('proto', 'val-stat', 'inclass_cnorm'),
]

result_guascor = [
    ('proto', 'train-stat', 'cor_mean'),
    ('proto', 'val-stat', 'cor_mean'),
    ('proto', 'test-stat', 'cor_mean'),
]
result_exclass = [
    ('proto', 'train-stat', 'est_dist'),
    ('proto', 'val-stat', 'est_dist'),
    ('proto', 'test-stat', 'est_dist')
]


if plot == 'protonet':
    plt.figure()
    plt.subplot(3,3,1)
    b.plot(result_db)
    plt.subplot(3,3,4)
    b.plot(result_acc)
    plt.subplot(3,3,2)
    b.plot(result_gaus)
    plt.subplot(3,3,5)
    b.plot(result_gaus3)
    plt.subplot(3,3,6)
    b.plot(result)
    plt.subplot(3,3,8)
    b.plot(result_guascor)
    plt.subplot(3,3,7)
    b.plot(result_inclass)
    plt.subplot(3,3,9)
    b.plot(result_exclass)
    plt.show()

#a.self_plot(result_dist_ratio)

#a.plot(result_dist)

print(a.get_log('dae', 'val', '5-way 5-shot val'))
