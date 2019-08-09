import tensorflow as tf
from model.loss import *
import tensorflow.contrib.slim as slim


def conv_block(inputs, out_channels, is_training=None, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.layers.batch_normalization(conv, training=is_training)
        #conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, 
        #    scale=True, center=True, is_training=is_training)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv

# Create model of CNN with slim api
def reg_CNN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(), #tf.truncated_normal_initializer(0.0, 0.01),
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


def four_block_cnn_encoder(x, h_dim, z_dim, is_training, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block(x, h_dim, is_training, name='conv_1')
        net = conv_block(net, h_dim, is_training, name='conv_2')
        net = conv_block(net, h_dim, is_training, name='conv_3')
        net = conv_block(net, z_dim, is_training, name='conv_4')
        net = tf.contrib.layers.flatten(net)
        print(net.get_shape()[-1])
        return net

def proto_model(support, query, ns, nq, k, label, metric='l2', center=None):
    # support :  [ns * k, d]
    # query   :  [nq * k, d]
    # label   :  [nq * k, k]
    z_dim = tf.shape(support)[1]
    # label
    label = tf.reshape(label, (-1, )) 
    qshape = tf.convert_to_tensor([nq, k, z_dim])
    sshape = tf.convert_to_tensor([ns, k, z_dim])

    proto = tf.reduce_mean(tf.reshape(support, sshape), axis=0)
    
    if metric == 'l2':
        logits = -euclidean_distance(query, proto)
    else:
        logits = cosine_similarity(query - center, proto - center)

    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(label, k), logits=logits, dim=1)
    entropy = tf.reduce_mean(entropy)
    #print(tf.argmax(logits, 1))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), label), tf.float32))   # [1
    return entropy, acc

