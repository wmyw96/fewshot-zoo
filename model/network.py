import tensorflow as tf


def conv_block(inputs, out_channels, is_training=None, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, 
            scale=True, center=True, is_training=is_training)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv

def four_block_cnn_encoder(x, h_dim, z_dim, is_training, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block(x, h_dim, name='conv_1')
        net = conv_block(net, h_dim, name='conv_2')
        net = conv_block(net, h_dim, name='conv_3')
        net = conv_block(net, z_dim, name='conv_4')
        net = tf.contrib.layers.flatten(net)
        return net

def proto_model(support, query, ns, nq, k, label):
    # support :  [ns * k, d]
    # query   :  [nq * k, d]
    # label   :  [nq * k, k]
    z_dim = tf.shape(support)[1]
    label_oneshot = tf.one_hot(label, depth=n_way)
    # label 
    qshape = tf.convert_to_tensor([nq, k, z_dim])
    sshape = tf.convert_to_tensor([ns, k, z_dim])

    proto = tf.reduce_mean(tf.reshape(support, sshape), axis=0)

    logits = -euclidean_distance(query, proto)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(tf.one_hot(label, k), logits, axis=1)
    entropy = tf.reduce_mean(entropy)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), label), tf.float32))   # [1
    return entropy, acc

