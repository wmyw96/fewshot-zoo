import tensorflow as tf
import numpy as np
from model.network import *
from model.layers import *
from model.network import *
from model.loss import *


def daie_pretrain_factory(inp, ph, params, reuse=True):
    if params['type'] == '4blockcnn':
        return reg_CNN(inp, ph['is_training'])
    elif params['type'] == 'resnet12':
        return resnet12(inp, ph)
    else:
        raise NotImplementedError


def daie_network_factory(inp, ph, params):
    if params['type'] == 'fc':
        return feedforward(inp, ph['is_training'], params, 'fc')
    else:
        raise ValueError('Not Impelmented')


def daie_disc_factory(inp, label, ph, params, return_inp=False):
    if params['type'] == 'fc':
        inp_concat = tf.concat([inp, label], axis=1)
        #if return_inp:
        out = feedforward(inp_concat, ph['is_training'], params, 'fc')
        if return_inp:
            return out, inp_concat
        else:
            return out
    else:
        raise ValueError('Not Impelmented')


##########################################
#
#      x  ---+
#            |
#            +--->  ex ---> classification
#                    ~ 
#            +--->  ez 
#            |
#      z  ---+
#  (gaussian)
###########################################

def get_daie_ph(params):
    ph = {}

    # data shape [b, w, h, c]
    # label shape [b, ]
    # lr decay
    # is_training

    # for training
    params_d = params['data']

    ph['data'] = tf.placeholder(dtype=tf.float32, 
                                shape=[None] + params_d['x_size'],
                                name='x')
    ph['label'] = tf.placeholder(dtype=tf.int64,
                                shape=[None],
                                name='label')

    ph['data_'] = tf.placeholder(dtype=tf.float32, 
                                shape=[None] + params_d['x_size'],
                                name='x')
    ph['label_'] = tf.placeholder(dtype=tf.int64,
                                shape=[None],
                                name='label')

    ph['g_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='g_lr_decay')
    ph['e_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='e_lr_decay')
    ph['d_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='d_lr_decay')
    ph['p_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='p_lr_decay')
    ph['is_training'] = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    ph['p_y_prior'] = tf.placeholder(dtype=tf.float32, shape=[params['data']['nclass'],], 
                                     name='p_y_prior')

    ph['ns'] = tf.placeholder(dtype=tf.int32, shape=[], name='ns')
    ph['nq'] = tf.placeholder(dtype=tf.int32, shape=[], name='nq')
    ph['n_way'] = tf.placeholder(dtype=tf.int32, shape=[], name='n_way')
    ph['eval_label'] = tf.placeholder(dtype=tf.int64,
                                shape=[None, None],
                                name='label')
    ph['stdw'] = tf.placeholder(dtype=tf.float32, shape=[], name='stdw')

    return ph

def get_daie_graph(params, ph):
    graph = {}
    if params['data']['dataset'] == 'mini-imagenet':
        rx = ph['data']
        with tf.variable_scope('pretrain'):
            graph['pt_logits'], graph['x'] = daie_pretrain_factory(rx, ph, params['pretrain'])
        x = graph['x']
        
    else:
        x = graph['x'] = ph['data']

    # !TODO: remove batch_size dependency
    batch_size = params['train']['batch_size']

    graph['one_hot_label'] = tf.one_hot(ph['label'], params['data']['nclass'])  # [b, K]
    graph['one_hot_label_'] = tf.one_hot(ph['label_'], params['data']['nclass'])  # [b, K]
    stddev = params['z']['stddev'] * ph['stdw']

    with tf.variable_scope('daie', reuse=False):
        # Encoder
        # Fake Samples
        with tf.variable_scope('x2e', reuse=False):
            graph['fake_e'] = daie_network_factory(x, ph, params['x2e'])

        # For evaluation
        ns, nq, n_way = ph['ns'], ph['nq'], ph['n_way']

        with tf.variable_scope('z-embed', reuse=False):
            if params['z']['type'] == 'gaussian':
                nclass = params['network']['nclass']
                z_dim = params['network']['z_dim']

                graph['mu'] = \
                    tf.get_variable('mu', [nclass, z_dim],
                                    initializer=tf.random_normal_initializer)
                print(graph['mu'])
                real_z_mean = tf.gather(graph['mu'], ph['label'], axis=0)
                noise = tf.random_normal([batch_size, z_dim], 0.0, stddev, 
                                         seed=params['train']['seed'])
                real_z = real_z_mean + noise
                graph['real_z'] = real_z
            else:
                raise ValueError('Not Implemented')

        with tf.variable_scope('z2e', reuse=False):
            graph['real_e'] = daie_network_factory(graph['real_z'], ph, params['z2e'])

        with tf.variable_scope('e2z', reuse=False):
            graph['real_z_hat'] = daie_network_factory(graph['real_e'], ph, params['e2z'])

        with tf.variable_scope('e2z', reuse=True):
            graph['z'] = graph['fake_z'] = daie_network_factory(graph['fake_e'], ph, params['e2z'])

        graph['embed'] = graph['mu']
        
        sz = graph['fake_z'][:ns*n_way,:]
        qz = graph['fake_z'][ns*n_way:,:]

        if params['network']['metric'] == 'l2':
            graph['eval_ent'], graph['eval_acc'] = proto_model(sz, qz, ns, nq, n_way, ph['eval_label'])
        else:
            nanase = tf.reduce_mean(graph['mu'], axis=0, keepdims=True)
            graph['eval_ent'], graph['eval_acc'] = proto_model(sz, qz, ns, nq, n_way, ph['eval_label'],
                                                               'cos', center=nanase)

        # Discriminator
        with tf.variable_scope('disc-e', reuse=False):
            graph['fake_e_critic'] = daie_disc_factory(graph['fake_e'], graph['one_hot_label'], 
                                                      ph, params['disc'])
        with tf.variable_scope('disc-e', reuse=True):
            graph['real_e_critic'] = daie_disc_factory(graph['real_e'], graph['one_hot_label'], 
                                                      ph, params['disc'])

        if params['disc']['gan-loss'] == 'wgan-gp':
            alpha = tf.random_uniform([batch_size, 1], 0, 1, 
                                      seed=params['train']['seed'])
            
            inds = tf.squeeze(tf.random_shuffle(tf.expand_dims(tf.range(batch_size), 1)))
            fake_e_, label_fake_e_ = graph['fake_e'], graph['one_hot_label']
            real_e_, label_real_e_ = tf.gather(graph['real_e'], inds), tf.gather(graph['one_hot_label'], inds)

            hat_e_ = fake_e_ * alpha + (1 - alpha) * real_e_
            label_hat_e_ = label_fake_e_ * alpha + (1 - alpha) * label_real_e_
            with tf.variable_scope('disc-e', reuse=True):
                graph['hat_e_critic'], graph['hat_e'] = \
                                        daie_disc_factory(hat_e_, label_hat_e_,
                                                         ph, params['disc'], True)

    return graph


def show_params(domain, var_list):
    print('Domain {}:'.format(domain))
    for var in var_list:
        print('{}: {}'.format(var.name, var.shape))

def get_daie_vars(params, ph, graph):
    pretrain_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrain')
    saved_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='daie')
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='daie/disc-e')
    x2e_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='daie/x2e')
    e2z_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='daie/z2e')
    z2e_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='daie/e2z')
    network_vars = x2e_vars + e2z_vars + z2e_vars
    z_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='daie/z-embed')

    graph_vars = {
        'disc': disc_vars,
        'x2e_vars': x2e_vars,
        'z2e_vars': z2e_vars,
        'e2z_vars': e2z_vars,
        'z_vars': z_vars,
        'gen': network_vars,
        'backbone': pretrain_vars
    }
    show_params('disc', disc_vars)
    show_params('gen', network_vars)
    show_params('z', z_vars)
    show_params('backbone', pretrain_vars)
    return graph_vars, saved_vars, pretrain_vars


def get_daie_targets(params, ph, graph, graph_vars):
    # disc loss
    if params['disc']['gan-loss'] == 'wgan-gp':
        w_dist = wgan_gp_wdist(graph['real_e_critic'], graph['fake_e_critic'])
        gradient_penalty = wgan_gp_gp_loss(graph['hat_e'], graph['hat_e_critic'])
        d_loss = -w_dist + params['disc']['gp_weight'] * gradient_penalty

        d_loss_all = d_loss #0.5 * (d_loss_inter + d_loss)
        disc_op = tf.train.AdamOptimizer(params['disc']['lr'] * ph['d_lr_decay'], beta1=0.5)
        disc_grads = disc_op.compute_gradients(loss=d_loss_all,
                                               var_list=graph_vars['disc'])
        disc_train_op = disc_op.apply_gradients(grads_and_vars=disc_grads)

        disc = {
            'train_op': disc_train_op,
            'w_dist_loss': w_dist,
            'gp_loss': gradient_penalty,
            'd_loss': d_loss,
            'd_loss_all': d_loss_all
        }
    elif params['disc']['gan-loss'] == 'gan':
        entropy = disc_gan_loss(graph['real_e_critic'], graph['fake_e_critic']) 
        #entropy = tf.log(tf.nn.sigmoid(graph['real_z_critic']) + 1e-9) + tf.log(1 - tf.nn.sigmoid(graph['fake_z_critic']) + 1e-9)
        disc_acc = 0.5 * (tf.reduce_mean(tf.to_float(tf.greater(graph['real_e_critic'], 0))) + \
			 tf.reduce_mean(tf.to_float(tf.less(graph['fake_e_critic'], 0))) )
        d_loss = entropy * params['disc']['em_weight']

        d_loss_all = d_loss
        disc_op = tf.train.AdamOptimizer(params['disc']['lr'] * ph['d_lr_decay'], beta1=0.5)
        disc_grads = disc_op.compute_gradients(loss=d_loss_all,
                                               var_list=graph_vars['disc'])
        disc_train_op = disc_op.apply_gradients(grads_and_vars=disc_grads)
        disc = {
            'train_op': disc_train_op,
            'entropy_loss': entropy,
            'disc_acc_loss': disc_acc,
            'd_loss': d_loss
        }
    else:
        raise ValueError('Not Implemented GAN loss')

    # network and embedding part

    gen = {}
    gen['g_loss'] = 0.0

    # embedding loss
    if params['disc']['gan-loss'] == 'wgan-gp':
        gen['embed_loss'] = w_dist #0.5 * (w_dist + w_dist_inter)
    elif params['disc']['gan-loss'] == 'gan':
        gen['embed_loss'] = -entropy

    gen['g_loss'] += gen['embed_loss'] * params['network']['e_m_weight']

    # reconstruction loss
    #if params['network']['use_decoder']:
    gen['rec_loss'] = tf.reduce_mean(tf.square(graph['real_z_hat'] - graph['real_z']))
    gen['g_loss'] += gen['rec_loss'] * params['network']['rec_weight']
    
    stddev = params['z']['stddev'] * ph['stdw']
    # classfication loss
    log_p_y_prior = tf.log(tf.expand_dims(ph['p_y_prior'], 0))      # [1, K]
    dist = euclidean_distance(graph['fake_z'], graph['mu'], scale=1.0/stddev/stddev)         # [b, K]
    
    logits = -dist + log_p_y_prior
    
    log_yz = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_label'], logits=logits, dim=1) # [b]
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), ph['label']), tf.float32))   # [1,]

    gen['cls_loss'] = tf.reduce_mean(log_yz)
    gen['acc_loss'] = acc
    gen['g_loss'] += gen['cls_loss'] * params['network']['cls_weight']

    if params['network']['update_pretrain']:
        pretrain_loss = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_label'], 
            logits=graph['pt_logits'], dim=1)
        pretrain_loss = tf.reduce_mean(pretrain_loss)
        pretrain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(graph['pt_logits'], 1), ph['label']), tf.float32))
        pretrain_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='pretrain')
        gen['update'] = pretrain_update_ops
        gen['pretrain_acc_loss'] = pretrain_acc
        gen['pretrain_ent_loss'] = pretrain_loss
        gen['g_loss'] += gen['pretrain_ent_loss'] * params['network']['pretrain_weight']

    gen_op = tf.train.AdamOptimizer(params['network']['lr'] * ph['g_lr_decay'], beta1=0.5)
    if not params['network']['update_pretrain']:
        gen_grads = gen_op.compute_gradients(loss=gen['g_loss'],
                                              var_list=graph_vars['gen'])
    else:
        gen_grads = gen_op.compute_gradients(loss=gen['g_loss'],
                                              var_list=graph_vars['gen'] + graph_vars['backbone'])

    gen_train_op = gen_op.apply_gradients(grads_and_vars=gen_grads)

    embed_op = tf.train.AdamOptimizer(params['z']['lr'] * ph['e_lr_decay'], beta1=0.5)
    embed_grads = embed_op.compute_gradients(loss=gen['g_loss'],
                                            var_list=graph_vars['z_vars'])
    embed_train_op = embed_op.apply_gradients(grads_and_vars=embed_grads) #, global_step=global_step_embed)

    gen['train_gen'] = gen_train_op
    gen['train_embed'] = embed_train_op

    if params['data']['dataset'] == 'mini-imagenet':
        pretrain_loss = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_label'], 
            logits=graph['pt_logits'], dim=1)
        pretrain_loss = tf.reduce_mean(pretrain_loss)
        pretrain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(graph['pt_logits'], 1), ph['label']), tf.float32))
        pretrain_op = tf.train.AdamOptimizer(params['pretrain']['lr'] * ph['p_lr_decay'])
        pretrain_grads = pretrain_op.compute_gradients(loss=pretrain_loss,
                                          var_list=graph_vars['backbone'])
        pretrain_train_op = pretrain_op.apply_gradients(grads_and_vars=pretrain_grads)
        pretrain_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='pretrain')
        pretrain = {
            'train': pretrain_train_op,
            'update': pretrain_update_ops,
            'acc': pretrain_acc,
        }
        pretrain_eval = {'acc': pretrain_acc}
    else:
        pretrain = {}
        pretrain_eval = {}

    targets = {
        'gen': gen,
        'disc': disc,
        'pretrain': pretrain,
        'pretrain_eval': pretrain_eval,
        'eval': {
            'acc': graph['eval_acc'],
            '64-acc': acc
        },
    }

    return targets

def build_daie_model(params):
    ph = get_daie_ph(params)

    graph = get_daie_graph(params, ph)
    graph_vars, saved_vars, pretrain_vars = get_daie_vars(params, ph, graph)
    targets = get_daie_targets(params, ph, graph, graph_vars)

    return ph, graph, targets, saved_vars, pretrain_vars

