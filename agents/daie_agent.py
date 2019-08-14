from model.daie import *
from agents.utils import *
import signal
import matplotlib.pyplot as plt
from agents import statistics as stat
from tqdm import tqdm


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class DAIE(object):
    def __init__(self, params, logger, gpu=-1):
        self.params = params

        # Build computation graph for the DDPG agent
        self.ph, self.graph, self.targets, self.save_vars, self.pretrain_vars = build_dae_model(params)
        self.gpu = gpu
        self.epoch = 0

        self.logger = logger
        # Session and saver
        if self.gpu > -1:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        else:
            self.sess = tf.Session()
        self.step = 0

        self.stdw = 1.0
        self.g_decay = 1.0
        self.d_decay = 1.0
        self.e_decay = 1.0
        self.best_valid = [0.0] * len(self.params['test']['shot'])
        self.test_perf = [0.0] * len(self.params['test']['shot'])

        self.losses = {}
        self.nclass = params['data']['nclass']
        self.save_model = tf.train.Saver(var_list=self.save_vars)
        if params['data']['dataset'] == 'mini-imagenet':
            self.save_pretrain = tf.train.Saver(var_list=self.pretrain_vars)
        self.killer = GracefulKiller()

    def start(self, data_loader=None, load_pretrain_dir=None):
        self.sess.run(tf.global_variables_initializer())
        accs = []
        if load_pretrain_dir is not None:
            self.save_pretrain.restore(self.sess, os.path.join(load_pretrain_dir, 'pretrain.ckpt'))
            batch_size = 400
            embed = [ [] for i in range(self.nclass) ]
            for it in range(self.params['pretrain']['iter_per_epoch']):
                inputs, labels = data_loader.next_batch(batch_size)
                fetch = self.sess.run(self.targets['pretrain_eval'],
                                      feed_dict={
                                        self.ph['data']: inputs,
                                        self.ph['label']: labels,
                                        self.ph['p_lr_decay']: 1.0,
                                        self.ph['is_training']: False,
                                        self.ph['p_y_prior']: data_loader.get_weight()
                                      })
                accs.append(fetch['acc'])
            print('Acc = {}'.format(np.mean(accs)))

    def pretrain(self, data_loader, save_dir):
        self.sess.run(tf.global_variables_initializer())
        batch_size = self.params['pretrain']['batch_size']
        for epoch in range(self.params['pretrain']['num_epoches']):
            accs = []
            for it in range(self.params['pretrain']['iter_per_epoch']):
                inputs, labels = data_loader.next_batch(batch_size)
                fetch = self.sess.run(self.targets['pretrain'],
                                      feed_dict={
                                        self.ph['data']: inputs,
                                        self.ph['label']: labels,
                                        self.ph['p_lr_decay']: 1.0,
                                        self.ph['is_training']: True,
                                        self.ph['p_y_prior']: data_loader.get_weight()
                                      })
                accs.append(fetch['acc'])
            print('Pretrain Epoch {}: Accuracy = {}'.format(epoch, np.mean(accs)))
            self.save_pretrain.save(self.sess, os.path.join(save_dir, 'pretrain.ckpt'))

    def train_iter(self, data_loader):
        n_critic = self.params['disc']['n_critic']
        batch_size = self.params['train']['batch_size']

        for i in range(n_critic):
            inputs, labels = data_loader.next_batch(batch_size)
            inputs_, labels_ = data_loader.next_batch(batch_size)

            fetch = self.sess.run(self.targets['disc'], 
                                  feed_dict={
                                    self.ph['data']: inputs,
                                    self.ph['label']: labels,
                                    self.ph['data_']: inputs_,
                                    self.ph['label_']: labels_,                                    
                                    self.ph['d_lr_decay']: self.d_decay,
                                    self.ph['is_training']: True,
                                    self.ph['stdw']: self.stdw,
                                    self.ph['p_y_prior']: data_loader.get_weight()
                                  })
            update_loss(fetch, self.losses)

        inputs, labels = data_loader.next_batch(batch_size)
        inputs_, labels_ = data_loader.next_batch(batch_size)
        fetch = self.sess.run(self.targets['gen'], 
                              feed_dict={
                                    self.ph['data']: inputs,
                                    self.ph['label']: labels,
                                    self.ph['data_']: inputs_,
                                    self.ph['label_']: labels_,
                                    self.ph['g_lr_decay']: self.g_decay,
                                    self.ph['e_lr_decay']: self.e_decay,
                                    self.ph['is_training']: True,
                                    self.ph['stdw']: self.stdw,
                                    self.ph['p_y_prior']: data_loader.get_weight()
                              })
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
        if 'adaptive_std' in self.params['embedding']:
            self.stdw -= self.params['embedding']['adaptive_std']['decay']  
            print('Stddev decay, current = {}'.format(self.stdw))      
        if self.epoch % self.params['network']['n_decay'] == 0:
            self.g_decay *= self.params['network']['weight_decay']
            print('G Decay, Current = {}'.format(self.g_decay))
        if self.epoch % self.params['disc']['n_decay'] == 0:
            self.d_decay *= self.params['disc']['weight_decay']
            print('D Decay, Current = {}'.format(self.d_decay))
        if self.epoch % self.params['embedding']['n_decay'] == 0:
            self.e_decay *= self.params['embedding']['weight_decay']
            print('E Decay, Current = {}'.format(self.e_decay))
        
    def print_log(self, epoch):
        print_log('DAiE Training: ', epoch, self.losses)
        self.logger.print(epoch, 'training', self.losses)
        self.losses = {}

    def single_eval(self, epoch, data_loader, n_way, shot):
        accs, acs = [], []
        for _ in range(self.params['test']['num_episodes']):
            support, query, labels, _b = \
                data_loader.get_support_query(n_way, shot, self.params['test']['nq'], True)
            flats = np.reshape(support, [n_way * shot] + self.params['data']['x_size'])
            flatq = np.reshape(query, [n_way * self.params['test']['nq']] + self.params['data']['x_size'])
            flat_inp = np.concatenate([flats, flatq], 0)
            acc = self.sess.run([self.targets['eval']['acc']], 
                                  feed_dict={
                                        self.ph['data']: flat_inp,
                                        self.ph['ns']: shot,
                                        self.ph['nq']: self.params['test']['nq'],
                                        self.ph['n_way']: n_way,
                                        self.ph['eval_label']: labels,
                                        self.ph['is_training']: False
                                  })
            #print(k)
            accs.append(acc)
        return np.mean(accs)
    
    def eval(self, epoch, valid, test):
        log_dict = {}

        for idx in range(len(self.params['test']['shot'])):
            shot = self.params['test']['shot'][idx]
            n_way = self.params['test']['n_way'][idx]
            cur_value = self.single_eval(epoch, valid, n_way, shot)
            if cur_value > self.best_valid[idx]:
                self.best_valid[idx] = cur_value
                self.test_perf[idx] = self.single_eval(epoch, test, n_way, shot)
            print('Epoch {}: [{}-way {}-shot] valid perf {}, '
                'test perf {}'.format(epoch, n_way, shot, cur_value, self.test_perf[idx]))

            cur_log = {'{}-way {}-shot val'.format(n_way, shot): cur_value,
                       '{}-way {}-shot test'.format(n_way, shot): self.test_perf[idx]}
            update_loss(cur_log, log_dict, False)
            print(log_dict)

        self.logger.print(epoch, 'val', log_dict)

    def get_statistics(self, epoch, domain, data_loader, col, batch_size=600):
        log_dict, embed = {}, None
        if domain == 'train':
            embed = self.sess.run(self.graph['embed'])
            update_loss(stat.norm(embed, 'embed_'), log_dict, False)
            update_loss(stat.pairwise_distance(embed, 'embed_'), log_dict, False)

        inp = []
        label = []
        embed2 = []
        dbis = []       
        for clsid in tqdm(range(data_loader.nclass), desc='Get Statistics {}'.format(domain)):
            x = data_loader.sample_from_class(clsid, batch_size)
            z = self.sess.run(self.graph['z'], feed_dict={self.ph['data']: x, self.ph['is_training']: False})
            update_loss(stat.gaussian_test(z), log_dict, False)
            update_loss(stat.correlation(z), log_dict, False)
            
            nanasa = np.mean(z, 0, keepdims=True)
            embed2.append(nanasa)
            dbis.append(np.sqrt(np.mean(np.sum(np.square(z - nanasa), 1), 0)))
            update_loss({'mean_std': np.mean(np.std(z, 0))}, log_dict, False)            
            update_loss(stat.norm(z[:100], 'inclass_'), log_dict, False)
            update_loss(stat.pairwise_distance(z[:100], 'inclass_'), log_dict, False)
            inp.append(z[:50, ])
            label += [clsid] * 50
        embed2 = np.concatenate(embed2, 0)
        if embed is not None:
            update_loss({'mean_div': np.sqrt(np.sum(np.square(embed - embed2))) / embed.shape[0]}, log_dict, False)
        update_loss(stat.norm(embed2, 'est_'), log_dict, False)
        update_loss(stat.pairwise_distance(embed2, 'est_'), log_dict, False)
        update_loss(stat.davies_bouldin_index(np.array(dbis), stat.l2_distance(embed2)), log_dict, False)

        inputs = np.concatenate(inp, axis=0)
        labels = np.array(label)
        #stat.tsne_visualization(inputs, labels, os.path.join(self.logger.dir, 
        #    'epoch{}_{}.png'.format(epoch, domain)), col)
        self.logger.print(epoch, domain + '-stat', log_dict)
        return self.terminate()

    def terminate(self):
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
