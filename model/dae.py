import tensorflow as tf
import numpy as np
from model.network import *
from model.layers import *
from model.network import *
from model.loss import *


def dae_encoder_factory(inp, ph, params, reuse=True):
    if params['type'] == 'fc':
        return feedforward(inp, ph['is_training'], params, 'fc')
    elif params['type'] == '4blockcnn':
        feat = four_block_cnn_encoder(inp, 64, 64, ph['is_training'])
        return feedforward(feat, ph['is_training'], params, 'fc')
    else:
        raise NotImplementedError


def dae_decoder_factory(inp, ph, params):
    if params['type'] == 'fc':
        return feedforward(inp, ph['is_training'], params, 'fc')
    else:
        raise ValueError('Not Impelmented')


def dae_disc_factory(inp, label, ph, params, return_inp=False):
    if params['type'] == 'fc':
        inp_vec = tf.get_variable('inp_vec', [params['nclass'], params['onehot_dim']], 
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        cat_vec = tf.matmul(label, inp_vec)
        inp_concat = tf.concat([inp, cat_vec], axis=1)
        #if return_inp:
        out = feedforward(inp_concat, ph['is_training'], params, 'fc')
        if return_inp:
            return out, inp
        else:
            return out
    else:
        raise ValueError('Not Impelmented')


def get_dae_ph(params):
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
    ph['d_lr_decay'] = tf.placeholder(dtype=tf.float32, shape=[], name='d_lr_decay')
    ph['is_training'] = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    ph['p_y_prior'] = tf.placeholder(dtype=tf.float32, shape=[params['data']['nclass'],], 
                                     name='p_y_prior')

    # for evaluation
    #ph['support'] = tf.placeholder(dtype=tf.float32, 
    #                               shape=[None, None] + params_d['x_size'],
    #                               name='s')
    #ph['query'] = tf.placeholder(dtype=tf.float32, 
    #                             shape=[None, None] + params_d['x_size'],
    #                             name='q')
    ph['ns'] = tf.placeholder(dtype=tf.int32, shape=[], name='ns')
    ph['nq'] = tf.placeholder(dtype=tf.int32, shape=[], name='nq')
    ph['n_way'] = tf.placeholder(dtype=tf.int32, shape=[], name='n_way')
    ph['eval_label'] = tf.placeholder(dtype=tf.int64,
                                shape=[None, None],
                                name='label')
    ph['stdw'] = tf.placeholder(dtype=tf.float32, shape=[], name='stdw')
    ph['cd_embed'] = tf.placeholder(dtype=tf.float32, shape=[params['data']['nclass'], params['network']['z_dim']], name='cd_embed')

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
            z = dae_encoder_factory(x, ph, params['encoder'], False)
            graph['z'] = z

        # For evaluation
        #ns = tf.shape(ph['support'])[0]
        #nq = tf.shape(ph['query'])[0]
        #n_way = tf.shape(ph['support'])[1]
        ns, nq, n_way = ph['ns'], ph['nq'], ph['n_way']
        
        #sx = tf.reshape(ph['support'], tf.convert_to_tensor([ns*n_way] + params['data']['x_size']))  # [ns * k, sz]
        #qx = tf.reshape(ph['query'], tf.convert_to_tensor([nq*n_way] + params['data']['x_size']))    # [nq * k, sz]

        #with tf.variable_scope('encoder', reuse=True):
        #    sz = graph['support_z'] = dae_encoder_factory(x[:ns*n_way,:], ph, params['encoder'])
        #with tf.variable_scope('encoder', reuse=True):
        #    qz = graph['query_z'] = dae_encoder_factory(x[ns*n_way:,:], ph, params['encoder'])
        #sz = z[:ns*n_way,:]
        #qz = z[ns*n_way:,:]
        #graph['eval_ent'], graph['eval_acc'] = proto_model(sz, qz, ns, nq, n_way, ph['eval_label'])
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
                stddev *= ph['stdw']
                real_z_mean = tf.gather(graph['mu'], ph['label'], axis=0)
                noise = tf.random_normal([batch_size, z_dim], 0.0, stddev, 
                                         seed=params['train']['seed'])
                real_z = real_z_mean + noise
                graph['real_z'] = real_z
                graph['fake_z'] = z
                
        graph['embed'] = graph['mu']
        if 'vmf' in params['network']:
            graph['fake_z'] = normalize(graph['fake_z'])
            graph['real_z'] = normalize(graph['real_z'])
            graph['embed'] = normalize(graph['mu'])
        
        sz = graph['z'][:ns*n_way,:]
        qz = graph['z'][ns*n_way:,:]

        if params['network']['metric'] == 'l2':
            graph['eval_ent'], graph['eval_acc'] = proto_model(sz, qz, ns, nq, n_way, ph['eval_label'])
        else:
            nanase = tf.reduce_mean(graph['mu'], axis=0, keepdims=True)
            graph['eval_ent'], graph['eval_acc'] = proto_model(sz, qz, ns, nq, n_way, ph['eval_label'],
                                                               'cos', center=nanase)
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
                graph['hat_z_critic'], graph['hat_z'] = \
                                        dae_disc_factory(graph['hat_z'], graph['one_hot_label'], 
                                                         ph, params['disc'], True)
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
    
    stddev = params['embedding']['stddev'] * ph['stdw']
    # classfication loss
    log_p_y_prior = tf.log(tf.expand_dims(ph['p_y_prior'], 0))      # [1, K]
    dist = euclidean_distance(graph['z'], graph['mu'], scale=1.0/stddev/stddev)         # [b, K]
    
    logits = -dist + log_p_y_prior
    
    if 'vmf' in params['network']:
        logits = inner_product(graph['z'], normalize(graph['mu'])) + log_p_y_prior

    log_yz = tf.nn.softmax_cross_entropy_with_logits(labels=graph['one_hot_label'], logits=logits, dim=1) # [b]
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), ph['label']), tf.float32))   # [1,]

    gen['cls_loss'] = tf.reduce_mean(log_yz)
    gen['acc_loss'] = acc
    gen['g_loss'] += gen['cls_loss'] * params['network']['cls_weight']

    # penalize the norm of the embedding    
    if 'l2' in params['embedding']:
        gen['embed_l2_loss'] = tf.reduce_mean(tf.square(graph['mu']))
        gen['g_loss'] += params['embedding']['l2'] * gen['embed_l2_loss']

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='dae/encoder')
    #print(update_ops)
    if len(update_ops) > 0:
        gen['update'] = update_ops

    gen_op = tf.train.AdamOptimizer(params['network']['lr'] * ph['g_lr_decay'])
    gen_grads = gen_op.compute_gradients(loss=gen['g_loss'],
                                          var_list=graph_vars['gen'])
    gen_train_op = gen_op.apply_gradients(grads_and_vars=gen_grads)

    #global_step_embed = tf.Variable(tf.constant(0))
    #embed_lr = tf.train.exponential_decay(0.01, global_step_embed, 
    #                                      int(80 * 2), 0.95, staircase=True)
    #embed_op = tf.train.GradientDescentOptimizer(embed_lr)
    embed_op = tf.train.AdamOptimizer(params['embedding']['lr'] * ph['e_lr_decay'])
    embed_grads = embed_op.compute_gradients(loss=gen['g_loss'],
                                            var_list=graph_vars['embed'])
    embed_train_op = embed_op.apply_gradients(grads_and_vars=embed_grads) #, global_step=global_step_embed)

    gen['train_gen'] = gen_train_op
    gen['train_embed'] = embed_train_op

    targets = {
        'gen': gen,
        'disc': disc,
        'eval': {
            'acc': graph['eval_acc'],
            '64-acc': acc
        },
        'assign_embed': {
            'assign': tf.assign(graph['mu'], ph['cd_embed'])
        }
    }

    return targets

def build_dae_model(params):
    ph = get_dae_ph(params)

    graph = get_dae_graph(params, ph)
    graph_vars, saved_vars = get_dae_vars(params, ph, graph)
    targets = get_dae_targets(params, ph, graph, graph_vars)

    return ph, graph, targets, saved_vars

