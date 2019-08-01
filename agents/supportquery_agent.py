from model.protonet import *
from agents.utils import *
import signal
import matplotlib.pyplot as plt
import time
import agents.statistics as stat
from tqdm import tqdm


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class SupportQueryAgent(object):
    def __init__(self, params, logger, gpu=-1):
        self.params = params
        self.logger = logger
        if params['network']['model'] == 'protonet':
            self.ph, self.graph, self.targets, self.save_vars = build_protonet_model(params)
        else:
            raise NotImplementedError
        self.gpu = gpu
        self.epoch = 0
        
        #print(gpu)
        # Session and saver
        if self.gpu > -1:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        else:
            self.sess = tf.Session()
        self.step = 0

        self.decay = 1.0
        self.best_valid = 0.0
        self.losses = {}
        self.nclass = params['data']['nclass']
        self.save_model = tf.train.Saver(var_list=self.save_vars)
        self.killer = GracefulKiller()

    def start(self):
        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, data_loader):
        dat_time = -time.time()
        support, query, labels = \
            data_loader.get_support_query(self.params['train']['n_way'],
                self.params['train']['shot'], self.params['train']['nq'])
        dat_time += time.time()
        
        #print(query.shape)
        #print(support.shape)
        fetch_time = -time.time()
        fetch = self.sess.run(self.targets['gen'], 
                              feed_dict={
                                    self.ph['support']: support,
                                    self.ph['query']: query,
                                    self.ph['label']: labels, 
                                    self.ph['lr_decay']: self.decay,
                                    self.ph['is_training']: True
                              })
        fetch_time += time.time()
        #print('Data Time = {}, Fetch Time = {}'.format(dat_time, fetch_time))
        update_loss(fetch, self.losses)
        self.step += 1

        if self.killer.kill_now:
            save_option = input('Save current model (y/[n])?')
            if save_option == 'y':
                print('Saved Successfully !')
                #self.saver_expert.save(self.sess, self.save_expert_dir+'/expert.ckpt')

            kill_option = input('Kill session (y/[n])?')
            if kill_option == 'y':
                self.sess.close()
                #self.writer.close()
                return True
            else:
                self.killer.kill_now = False
        return False

    def take_step(self):
        self.epoch += 1
        if self.epoch % self.params['network']['n_decay'] == 0:
            self.decay *= self.params['network']['weight_decay']
            print('Decay, Current = {}'.format(self.decay))
    
    def single_eval(self, epoch, data_loader):
        accs = []
        for _ in range(self.params['test']['num_episodes']):
            support, query, labels = \
                data_loader.get_support_query(self.params['test']['n_way'],
                    self.params['test']['shot'], self.params['test']['nq'])
            acc, k = self.sess.run([self.targets['eval']['acc'], self.graph['n_way']], 
                                  feed_dict={
                                        self.ph['support']: support,
                                        self.ph['query']: query,
                                        self.ph['label']: labels,
                                        self.ph['is_training']: False
                                  })
            #print(k)
            accs.append(acc)
        return np.mean(accs)
        #print('Valid Performance = {}'.format(np.mean(accs)))
    
    def eval(self, epoch, valid, test):
        cur_value = self.single_eval(epoch, valid)
        print('Epoch {}: valid performance {}'.format(epoch, cur_value))
        if cur_value > self.best_valid:
            self.best_valid = cur_value
            self.test_perf = self.single_eval(epoch, test)
        print('Current test performance {}'.format(self.test_perf))
        loss = {'valid_acc': [cur_value], 'test_acc': [self.test_perf]}
        self.logger.print(epoch, 'valid', loss)

    def print_log(self, epoch):
        print_log('{} training: '.format(self.params['network']['model']), epoch, self.losses)
        self.logger.print(epoch, 'training', self.losses)
        self.losses = {}

    def get_statistics(self, epoch, domain, data_loader, col, batch_size=100):
        log_dict, embed = {}, None

        inp = []
        label = []
        embed2 = []
        for clsid in tqdm(range(data_loader.nclass), desc='Get Statistics {}'.format(domain)):
            x = data_loader.sample_from_class(clsid, batch_size)
            x = np.expand_dims(x, 0)
            z = self.sess.run(self.graph['support_z'], feed_dict={self.ph['support']: x, self.ph['query']: x[:, :2, ], self.ph['is_training']: False})
            z = z.squeeze()
            update_loss(stat.gaussian_test(z), log_dict, False)
            update_loss(stat.correlation(z), log_dict, False)

            nanasa = np.mean(z, 0, keepdims=True)
            embed2.append(nanasa)
            update_loss(stat.norm(z, 'inclass_'), log_dict, False)
            update_loss(stat.pairwise_distance(z, 'inclass_'), log_dict, False)
            inp.append(z[:50, ])
            label += [clsid] * 50
        
        embed2 = np.concatenate(embed2, 0)
        update_loss(stat.norm(embed2, 'est_'), log_dict, False)
        update_loss(stat.pairwise_distance(embed2, 'est_'), log_dict, False)

        inputs = np.concatenate(inp, axis=0)
        labels = np.array(label)
        stat.tsne_visualization(inputs, labels, os.path.join(self.logger.dir,
            'epoch{}_{}.png'.format(epoch, domain)), col)
        self.logger.print(epoch, domain + '-stat', log_dict)
