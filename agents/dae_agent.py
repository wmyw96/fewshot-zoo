from model.dae import *
from agents.utils import *
import signal


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

        # Session and saver
        if self.gpu > -1:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        else:
            self.sess = tf.Session()
        self.step = 0
        self.decay = 1.0
        self.losses = {}

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
                                    self.ph['lr_decay']: self.decay,
                                    self.ph['is_training']: True,
                                    self.ph['p_y_prior']: data_loader.get_weight()
                                  })
            update_loss(fetch, self.losses)

        inputs, labels = data_loader.next_batch(batch_size)
        fetch = self.sess.run(self.targets['disc'], 
                              feed_dict={
                                    self.ph['data']: inputs,
                                    self.ph['label']: labels,
                                    self.ph['lr_decay']: self.decay,
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

    def print_log(self, epoch):
        print('DAE Training: ', epoch, self.losses)
        self.losses = {}
