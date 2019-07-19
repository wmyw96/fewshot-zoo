from model.protonet import *
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


class SupportQueryAgent(object):
    def __init__(self, params, gpu=-1):
        self.params = params

        if params['network']['model'] == 'protonet':
            self.ph, self.graph, self.targets, self.save_vars = build_dae_model(params)
        else:
            raise NotImplementedError
        self.gpu = gpu
        self.epoch = 0

        # Session and saver
        if self.gpu > -1:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        else:
            self.sess = tf.Session()
        self.step = 0

        self.decay = 1.0

        self.losses = {}
        self.nclass = params['data']['nclass']
        self.save_model = tf.train.Saver(var_list=self.save_vars)
        self.killer = GracefulKiller()

    def start(self):
        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, data_loader):
        support, query, labels = \
            data_loader.get_support_query(self.params['train']['n_way'],
                self.params['train']['shot'], self.params['train']['nq'])
        fetch = self.sess.run(self.targets['gen'], 
                              feed_dict={
                                    self.ph['support']: inputs,
                                    self.ph['query']: labels,
                                    self.ph['lr_decay']: self.decay,
                                    self.ph['is_training']: True
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
        if self.epoch % self.params['network']['n_decay'] == 0:
            self.g_decay *= self.params['network']['weight_decay']
            print('Decay, Current = {}'.format(self.decay))
    
    def eval(self, data_loader):
        accs = []
        for _ in range(self.params['test']['num_episodes']):
            support, query, labels = \
                data_loader.get_support_query(self.params['test']['n_way'],
                    self.params['test']['shot'], self.params['test']['nq'])
            acc = self.sess.run(self.targets['gen']['acc'], 
                                  feed_dict={
                                        self.ph['support']: inputs,
                                        self.ph['query']: labels,
                                        self.ph['is_training']: False
                                  })
            accs.append(acc)
        print('Valid Performance = {}'.format(np.mean(accs)))

    def print_log(self, epoch):
        print_log('{} training: '.format(params['network']['model']), epoch, self.losses)
        self.losses = {}
