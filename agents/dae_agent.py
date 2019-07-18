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
        self.decay = 1.0
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
                                    self.ph['lr_decay']: self.decay,
                                    self.ph['is_training']: True,
                                    self.ph['p_y_prior']: data_loader.get_weight()
                                  })
            update_loss(fetch, self.losses)

        inputs, labels = data_loader.next_batch(batch_size)
        fetch = self.sess.run(self.targets['gen'], 
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
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)
        plt.savefig(patho)
        plt.close()
    
    def take_step(self):
        self.epoch += 1
        
    def print_log(self, epoch):
        print_log('DAE Training: ', epoch, self.losses)
        self.losses = {}
