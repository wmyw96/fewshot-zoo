from model.dae import *
from agents.utils import *
import signal
import matplotlib.pyplot as plt


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class DAE(object):
    def __init__(self, params, gpu=-1):
        self.params = params

        # Build computation graph for the DDPG agent
        self.ph, self.graph, self.targets, self.save_vars = build_dae_model(params)
        self.gpu = gpu
        self.epoch = 0

        # Session and saver
        if self.gpu > -1:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        else:
            self.sess = tf.Session()
        self.step = 0

        self.g_decay = 1.0
        self.d_decay = 1.0
        self.e_decay = 1.0
        self.best_valid = [0.0] * len(self.params['test']['shot'])
        self.test_perf = [0.0] * len(self.params['test']['shot'])

        self.losses = {}
        self.nclass = params['data']['nclass']
        self.save_model = tf.train.Saver(var_list=self.save_vars)
        self.killer = GracefulKiller()

    def start(self):
        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, data_loader):
        n_critic = self.params['disc']['n_critic']
        batch_size = self.params['train']['batch_size']

        for i in range(n_critic):
            inputs, labels = data_loader.next_batch(batch_size)
            fetch = self.sess.run(self.targets['disc'], 
                                  feed_dict={
                                    self.ph['data']: inputs,
                                    self.ph['label']: labels,
                                    self.ph['d_lr_decay']: self.d_decay,
                                    self.ph['is_training']: False,
                                    self.ph['p_y_prior']: data_loader.get_weight()
                                  })
            update_loss(fetch, self.losses)

        inputs, labels = data_loader.next_batch(batch_size)
        fetch = self.sess.run(self.targets['gen'], 
                              feed_dict={
                                    self.ph['data']: inputs,
                                    self.ph['label']: labels,
                                    self.ph['g_lr_decay']: self.g_decay,
                                    self.ph['e_lr_decay']: self.e_decay,
                                    self.ph['is_training']: True,
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
    
    def visualize2d(self, path, data_loader, epoch, color):
        x, y = data_loader.next_batch(5000)
        plt.figure(figsize=(8, 8))
        embed = self.sess.run(self.graph['mu'])
        fake_z = self.sess.run(self.graph['fake_z'], feed_dict={self.ph['data']:x, self.ph['is_training']: False})
        
        for i in range(self.nclass):
            # samples
            #plt.scatter(real[i, :, 0], real[i, :, 1], c=color[i], s=0.3, marker='*')
            point = embed[i]
            plt.scatter(point[0], point[1], c=color[i], s=20.0, marker='x')
        
        for i in range(x.shape[0]):
            plt.scatter(fake_z[i, 0], fake_z[i, 1], c=color[int(y[i])], s=0.3, marker='*')

        patho = '{}/epoch{}.png'.format(path, epoch)
        #plt.xlim(-8, 8)
        #plt.ylim(-8, 8)
        plt.savefig(patho)
        plt.close()
    
    def take_step(self):
        self.epoch += 1
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
        print_log('DAE Training: ', epoch, self.losses)
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

    '''def evallll(self, data_loader):
        return None
        accs = []
        for _ in range(self.params['test']['num_episodes']):
            data, label = data_loader.next_batch(100)
            acc = self.sess.run([self.targets['eval']['64-acc']],
                                  feed_dict={
                                        self.ph['data']: data,
                                        self.ph['p_y_prior']: data_loader.get_weight(),
                                        self.ph['label']: label,
                                        self.ph['is_training']: False
                                  })
            accs.append(acc)

        print('Evalllll Acc = {}'.format(np.mean(accs)))'''
    
    def eval(self, epoch, valid, test):
        for idx in range(len(self.params['test']['shot'])):
            shot = self.params['test']['shot'][idx]
            n_way = self.params['test']['n_way'][idx]
            cur_value = self.single_eval(epoch, valid, n_way, shot)
            if cur_value > self.best_valid[idx]:
                self.best_valid[idx] = cur_value
                self.test_perf[idx] = self.single_eval(epoch, test, n_way, shot)
            print('Epoch {}: [{}-way {}-shot] valid perf {}, '
                'test perf {}'.format(epoch, n_way, shot, cur_value, self.test_perf[idx]))

