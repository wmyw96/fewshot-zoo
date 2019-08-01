import tensorflow as tf
import numpy as np
from model.network import *
from model.layers import *
from model.network import *
from model.loss import *
import tensorflow.contrib.slim as slim


def dve_encoder_factory(inp, ph, params, reuse=True):
    if params['type'] == 'fc':
        pout = feedforward(inp, ph['is_training'], params, 'fc')
        dim = pout.get_shape()[-1] // 2
        print('Encoding out dim {}'.format(dim))
        return pout[:, :dim], pout[:, dim:]
    else:
        raise NotImplementedError


def dve_decoder_factory(inp, ph, params):
    if params['type'] == 'fc':
        return feedforward(inp, ph['is_training'], params, 'fc')
    else:
        raise ValueError('Not Impelmented')


#def regularized_pretrain_network(inp, ph):
#import tensorflow.contrib.slim as slim


# Create model of CNN with slim api
def reg_CNN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        nf = 64
        x = tf.reshape(inputs, [-1, 84, 84, 3])
        net = slim.conv2d(x, nf, [3, 3], scope='conv1', padding='SAME')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, nf, [3, 3], scope='conv2', padding='SAME')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.conv2d(net, nf, [3, 3], scope='conv3', padding='SAME')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.conv2d(net, nf, [3, 3], scope='conv4', padding='SAME')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.flatten(net, scope='flatten')
        z = tf.identity(net)
        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, is_training=is_training, scope='dropout1')  # 0.5 by default
        outputs = slim.fully_connected(net, 64, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs, z


def regularized_pretrain_network(inp, ph):
    return reg_CNN(inp, ph['is_training'])


def dve_pretrain_encoder_factory(inp, ph):
    return four_block_cnn_encoder(inp, 64, 64, ph['is_training'])


def dve_pretrain_decoder_factory(inp, ph):
    nf = 64
    lx_z = tf.reshape(in_z, [-1, 5, 5, nf])    
    lx_z = tf.layers.conv2d_transpose(lx_z, nf, 3, strides=(2, 2),
                                      padding='same', use_bias=False)   #[10, 10]
    lx_z = tf.nn.relu(lx_z)
    lx_z = tf.layers.conv2d_transpose(lx_z, nf, 3, strides=(2, 2),
                                      padding='same', use_bias=False)   #[20, 20]
    lx_z = tf.nn.relu(lx_z)
    lx_z = tf.layers.conv2d_transpose(lx_z, nf, 3, strides=(2, 2),
                                      padding='same', use_bias=False)   #[40, 40]
    lx_z = tf.nn.relu(lx_z)
    lx_z = tf.layers.conv2d_transpose(lx_z, nf, 3, use_bias=False)      #[42, 42]
    lx_z = tf.nn.relu(lx_z)
    lx_z = tf.layers.conv2d_transpose(lx_z, 3, 3, strides=(2, 2),
                                      padding='same', use_bias=False)   #[84, 84]
    return lx_z

def get_dve_ph(params):
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
    ph['g_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='g_lr_decay')
    ph['e_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='e_lr_decay')
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
    ph['cd_embed'] = tf.placeholder(dtype=tf.float32, shape=[params['data']['nclass'], params['network']['z_dim']], name='cd_embed')

    return ph

def get_dve_graph(params, ph):
    graph = {}
    if params['data']['dataset'] == 'mini-imagenet':
        rx = ph['data']            # [b, *x.shape]
        #with tf.variable_scope('pretrain'):
            #graph['x'], graph['pt_logits'] = regularized_pretrain_network(rx, ph)
        #    x = dve_pretrain_encoder_factory(rx, ph)
        #    graph['x'] = tf.layers.batch_normalization(x, training=ph['is_training'])
        #    fc = tf.layers.dense(graph['x'], 1024, activation=tf.nn.relu)
        #    graph['pt_logits'] = tf.layers.dense(fc, params['data']['nclass'], activation=None)
        with tf.variable_scope('pretrain'):
            graph['pt_logits'], graph['x'] = regularized_pretrain_network(rx, ph)
        x = graph['x']
        
    else:
        x = graph['x'] = ph['data']
    
    z_dim = params['network']['z_dim']

    graph['one_hot_label'] = tf.one_hot(ph['label'], params['data']['nclass'])  # [b, K]
    with tf.variable_scope('dve', reuse=False):
        # Encoder
        with tf.variable_scope('encoder', reuse=False):
            mu_z, sigma_z = dve_encoder_factory(x, ph, params['encoder'], False)
            #if params['network']['fixed']:
            #    log_sigma_sq_z = tf.zeros(tf.shape(mu_z))
            graph['mu_z'], graph['sigma_z'] = mu_z, sigma_z
            noise = tf.random_normal(tf.shape(mu_z), 0.0, 1.0, 
                                     seed=params['train']['seed'])
            z = mu_z + tf.exp(0.5 * sigma_z) * noise
            graph['z'] = z

        with tf.variable_scope('embedding', reuse=False):
            if params['embedding']['type'] == 'gaussian':
                nclass = params['network']['nclass']

                graph['mu'] = \
                    tf.get_variable('mu', [nclass, z_dim],
                                    initializer=tf.random_normal_initializer)

                graph['gt_mu'] = tf.gather(graph['mu'], ph['label'], axis=0)


        ns, nq, n_way = ph['ns'], ph['nq'], ph['n_way']
        
        sz = mu_z[:ns*n_way,:]
        qz = mu_z[ns*n_way:,:]
        if params['network']['metric'] == 'l2':
            graph['eval_ent'], graph['eval_acc'] = proto_model(sz, qz, ns, nq, n_way, ph['eval_label'])
        else:
            nanase = tf.reduce_mean(graph['mu'], axis=0, keepdims=True)
            graph['eval_ent'], graph['eval_acc'] = proto_model(sz, qz, ns, nq, n_way, ph['eval_label'],
                                                               'cos', center=nanase)
        # Decoder
        with tf.variable_scope('decoder', reuse=False):
            if params['network']['use_decoder']:
                x_rec = dve_decoder_factory(z, ph, params['decoder'])
                graph['x_rec'] = x_rec

    graph['step_embed'] = tf.Variable(tf.constant(0))
    graph['step_gen'] = tf.Variable(tf.constant(0))
    return graph


def show_params(domain, var_list):
    print('Domain {}:'.format(domain))
    for var in var_list:
        print('{}: {}'.format(var.name, var.shape))

def get_dve_vars(params, ph, graph):
    pretrain_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrain')
    saved_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dve')
    encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dve/encoder')
    decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dve/decoder')
    network_vars = encoder_vars + decoder_vars
    embed_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dve/embed')

    graph_vars = {
        'pretrain': pretrain_vars,
        'encoder': encoder_vars,
        'decoder': decoder_vars,
        'embed': embed_vars,
        'gen': network_vars
    }
    show_params('pretrain', pretrain_vars)
    show_params('gen', network_vars)
    show_params('embed', embed_vars)
    return graph_vars, saved_vars, pretrain_vars


def get_dve_targets(params, ph, graph, graph_vars):
    # network and embedding part

    gen = {}
    gen['g_loss'] = 0.0

    # embedding loss
    kl, mu_d = kl_divergence(graph['mu_z'], graph['sigma_z'], graph['gt_mu'])
    gen['embed_loss'] = tf.reduce_mean(kl)
    gen['distance_loss'] = tf.reduce_mean(mu_d)

    gen['g_loss'] += gen['embed_loss'] * params['network']['e_m_weight']

    # reconstruction loss
    if params['network']['use_decoder']:
        gen['rec_loss'] = tf.reduce_mean(tf.reduce_sum(tf.square(graph['x'] - graph['x_rec']), 1))
        gen['g_loss'] += gen['rec_loss'] * params['network']['rec_weight']

    # classfication loss
    log_p_y_prior = tf.log(tf.expand_dims(ph['p_y_prior'], 0))      # [1, K]
    dist = euclidean_distance(graph['z'], graph['mu']) / 2.0       # [b, K]

    logits = -dist #+ log_p_y_prior

    log_yz = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_label'], logits=logits, dim=1) # [b]
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), ph['label']), tf.float32))   # [1,]

    gen['cls_loss'] = tf.reduce_mean(log_yz)
    gen['acc_loss'] = acc
    gen['g_loss'] += gen['cls_loss'] * params['network']['cls_weight']
    

    gen_lr = tf.train.exponential_decay(params['network']['lr'], graph['step_gen'], params['network']['n_decay'], 
                                        params['network']['decay_weight'], staircase=True)
    gen_op = tf.train.GradientDescentOptimizer(gen_lr)
    gen_grads = gen_op.compute_gradients(loss=gen['g_loss'],
                                          var_list=graph_vars['gen'])
    gen_train_op = gen_op.apply_gradients(grads_and_vars=gen_grads, global_step=graph['step_gen'])

    embed_lr = tf.train.exponential_decay(params['embedding']['lr'], graph['step_embed'], params['embedding']['n_decay'],
                                          params['embedding']['decay_weight'], staircase=True)
    embed_op = tf.train.GradientDescentOptimizer(embed_lr)
    embed_grads = embed_op.compute_gradients(loss=gen['g_loss'],
                                            var_list=graph_vars['embed'])
    embed_train_op = embed_op.apply_gradients(grads_and_vars=embed_grads, global_step=graph['step_embed'])

    gen['train_gen'] = gen_train_op
    gen['train_embed'] = embed_train_op

    if params['data']['dataset'] == 'mini-imagenet':
        pretrain_loss = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_label'], 
            logits=graph['pt_logits'], dim=1)
        pretrain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(graph['pt_logits'], 1), ph['label']), tf.float32))
        pretrain_op = tf.train.AdamOptimizer(params['pretrain']['lr'] * ph['p_lr_decay'])
        pretrain_grads = pretrain_op.compute_gradients(loss=pretrain_loss,
                                          var_list=graph_vars['pretrain'])
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
        'pretrain': pretrain,
        'pretrain_eval': pretrain_eval,
        'gen': gen,
        'eval': {
            'acc': graph['eval_acc'],
        },
        'assign_embed': {
            'assign': tf.assign(graph['mu'], ph['cd_embed'])
        }
    }

    return targets

def build_dve_model(params):
    ph = get_dve_ph(params)

    graph = get_dve_graph(params, ph)
    graph_vars, saved_vars, pretrain_vars = get_dve_vars(params, ph, graph)
    targets = get_dve_targets(params, ph, graph, graph_vars)

    return ph, graph, targets, saved_vars, pretrain_vars

