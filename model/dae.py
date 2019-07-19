import tensorflow as tf
import numpy as np
from model.network import *
from model.layers import *
from model.network import *
from model.loss import *


def dae_encoder_factory(inp, ph, params):
    if params['type'] == 'fc':
        return feedforward(inp, ph['is_training'], params, 'fc')
    else:
        raise ValueError('Not Impelmented')


def dae_decoder_factory(inp, ph, params):
    if params['type'] == 'fc':
        return feedforward(inp, ph['is_training'], params, 'fc')
    else:
        raise ValueError('Not Impelmented')


def dae_disc_factory(inp, label, ph, params):
    if params['type'] == 'fc':
        inp_vec = tf.get_variable('inp_vec', [params['nclass'], params['onehot_dim']], 
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        cat_vec = tf.matmul(label, inp_vec)
        inp_concat = tf.concat([inp, cat_vec], axis=1)
        return feedforward(inp_concat, ph['is_training'], params, 'fc')
    else:
        raise ValueError('Not Impelmented')


def get_dae_ph(params):
    ph = {}

    # data shape [b, w, h, c]
    # label shape [b, ]
    # lr decay
    # is_training

    params_d = params['data']

    ph['data'] = tf.placeholder(dtype=tf.float32, 
                                shape=[None] + params_d['x_size'],
                                name='x')
    ph['label'] = tf.placeholder(dtype=tf.int64,
                                shape=[None],
                                name='label')
    ph['g_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='g_lr_decay')
    ph['e_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='e_lr_decay')
    ph['d_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='d_lr_decay')
    ph['is_training'] = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    ph['p_y_prior'] = tf.placeholder(dtype=tf.float32, shape=[params['data']['nclass'],], 
                                     name='is_training')

    return ph

def get_dae_graph(params, ph):
    graph = {}
    x = ph['data']            # [b, *x.shape]
    # !TODO: remove batch_size dependency
    batch_size = params['train']['batch_size']

    graph['one_hot_label'] = tf.one_hot(ph['label'], params['data']['nclass'])  # [b, K]
    with tf.variable_scope('dae', reuse=False):
        # Encoder
        # Fake Samples
        with tf.variable_scope('encoder', reuse=False):
            z = dae_encoder_factory(x, ph, params['encoder'])
            graph['fake_z'] = z

        # Decoder
        with tf.variable_scope('decoder', reuse=False):
            if params['network']['use_decoder']:
                x_rec = dae_decoder_factory(z, ph, params['decoder'])
                graph['x_rec'] = x_rec

        # Embedding
        # Real Samples
        with tf.variable_scope('embedding', reuse=False):
            if params['embedding']['type'] == 'gaussian':
                nclass = params['network']['nclass']
                z_dim = params['network']['z_dim']
                stddev = params['embedding']['stddev']

                graph['mu'] = \
                    tf.get_variable('mu', [nclass, z_dim],
                                    initializer=tf.random_normal_initializer)

                real_z_mean = tf.gather(graph['mu'], ph['label'], axis=0)
                noise = tf.random_normal([batch_size, z_dim], 0.0, stddev, 
                                         seed=params['train']['seed'])
                real_z = real_z_mean + noise
                graph['real_z'] = real_z
            else:
                raise ValueError('Not Implemented Embedding Type')

        # Discriminator
        with tf.variable_scope('disc-embed', reuse=False):
            graph['fake_z_critic'] = dae_disc_factory(graph['fake_z'], graph['one_hot_label'], 
                                                      ph, params['disc'])
        with tf.variable_scope('disc-embed', reuse=True):
            graph['real_z_critic'] = dae_disc_factory(graph['real_z'], graph['one_hot_label'], 
                                                      ph, params['disc'])

        if params['disc']['gan-loss'] == 'wgan-gp':
            alpha = tf.random_uniform([batch_size, 1], 0, 1, 
                                      seed=params['train']['seed'])
            graph['hat_z'] = real_z * alpha + (1 - alpha) * z
            with tf.variable_scope('disc-embed', reuse=True):
                graph['hat_z_critic'] = dae_disc_factory(graph['hat_z'], graph['one_hot_label'], 
                                                         ph, params['disc'])
    return graph


def show_params(domain, var_list):
    print('Domain {}:'.format(domain))
    for var in var_list:
        print('{}: {}'.format(var.name, var.shape))

def get_dae_vars(params, ph, graph):
    saved_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dae')
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dae/disc')
    encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dae/encoder')
    decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dae/decoder')
    network_vars = encoder_vars + decoder_vars
    embed_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dae/embed')

    graph_vars = {
        'disc': disc_vars,
        'encoder': encoder_vars,
        'decoder': decoder_vars,
        'embed': embed_vars,
        'gen': network_vars
    }
    show_params('disc', disc_vars)
    show_params('gen', network_vars)
    show_params('embed', embed_vars)
    return graph_vars, saved_vars


def get_dae_targets(params, ph, graph, graph_vars):
    # disc loss
    if params['disc']['gan-loss'] == 'wgan-gp':
        w_dist = wgan_gp_wdist(graph['real_z_critic'], graph['fake_z_critic'])
        gradient_penalty = wgan_gp_gp_loss(graph['hat_z'], graph['hat_z_critic'])
        d_loss = -w_dist + params['disc']['gp_weight'] * gradient_penalty

        disc_op = tf.train.AdamOptimizer(params['disc']['lr'] * ph['d_lr_decay'])
        disc_grads = disc_op.compute_gradients(loss=d_loss,
                                               var_list=graph_vars['disc'])
        disc_train_op = disc_op.apply_gradients(grads_and_vars=disc_grads)

        disc = {
            'train_op': disc_train_op,
            'w_dist_loss': w_dist,
            'gp_loss': gradient_penalty,
            'd_loss': d_loss
        }
    else:
        raise ValueError('Not Implemented GAN loss')

    # network and embedding part

    gen = {}
    gen['g_loss'] = 0.0

    # embedding loss
    gen['embed_loss'] = w_dist
    gen['g_loss'] += gen['embed_loss'] * params['network']['e_m_weight']

    # reconstruction loss
    if params['network']['use_decoder']:
        gen['rec_loss'] = tf.reduce_mean(tf.abs(ph['data'] - graph['x_rec']))
        gen['g_loss'] += gen['rec_loss'] * params['network']['rec_weight']

    # classfication loss
    log_p_y_prior = tf.log(tf.expand_dims(ph['p_y_prior'], 0))      # [1, K]
    dist = euclidean_distance(graph['fake_z'], graph['mu'])         # [b, K]

    logits = -dist + log_p_y_prior

    log_yz = tf.nn.softmax_cross_entropy_with_logits_v2(graph['one_hot_label'], logits, axis=1) # [b]
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), ph['label']), tf.float32))   # [1,]

    gen['cls_loss'] = tf.reduce_mean(log_yz)
    gen['acc_loss'] = acc
    gen['g_loss'] += gen['cls_loss'] * params['network']['cls_weight']

    gen_op = tf.train.AdamOptimizer(params['network']['lr'] * ph['g_lr_decay'])
    gen_grads = gen_op.compute_gradients(loss=gen['g_loss'],
                                          var_list=graph_vars['gen'])
    gen_train_op = gen_op.apply_gradients(grads_and_vars=gen_grads)

    embed_op = tf.train.AdamOptimizer(params['embedding']['lr'] * ph['e_lr_decay'])
    embed_grads = embed_op.compute_gradients(loss=gen['g_loss'],
                                            var_list=graph_vars['embed'])
    embed_train_op = embed_op.apply_gradients(grads_and_vars=embed_grads)

    gen['train_gen'] = gen_train_op
    gen['train_embed'] = embed_train_op

    targets = {
        'gen': gen,
        'disc': disc
    }

    return targets

def build_dae_model(params):
    ph = get_dae_ph(params)

    graph = get_dae_graph(params, ph)
    graph_vars, saved_vars = get_dae_vars(params, ph, graph)
    targets = get_dae_targets(params, ph, graph, graph_vars)

    return ph, graph, targets, saved_vars
