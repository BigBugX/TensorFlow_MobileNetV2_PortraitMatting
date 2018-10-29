__author__ = 'Will@PCVG'
# Utils used with tensorflow implemetation

import tensorflow as tf

weight_decay = 1e-4

def relu(x, name='relu6'):
	return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
	return tf.layers.batch_normalization(x,
						momentum = momentum,
						epsilon = epsilon,
						scale = True,
						training=train,
						name=name)


def conv2d(input_, w, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def conv2d_expt(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_block(input, w, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, w, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net


def get_variable_oct_dl(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def conv2d_head_oct_dl(inputs,
                        conv_w,
                        bn_w, 
                        bn_b, 
                        bn_m, 
                        bn_v,
                        strides=1,
                        is_train=True,
                        name=None,
                        padding='SAME',
                        bias=None,
                        eps=1e-5):
    with tf.variable_scope(name):
        #$ conv_w = get_variable_oct_dl(conv_w, 'conv_w')
        #$ bn_w = get_variable_oct_dl(bn_w, 'bn_w')
        #$ bn_b = get_variable_oct_dl(bn_b, 'bn_b')
        #$ bn_m = get_variable_oct_dl(bn_m, 'bn_m')
        #$ bn_v = get_variable_oct_dl(bn_v, 'bn_v')

        conv = tf.nn.conv2d(inputs, conv_w, strides=[1, strides, strides, 1], name='conv', padding=padding)
        if bias:
            conv = tf.nn.bias_add(conv, bias)
        conv = tf.nn.batch_normalization(conv, bn_m, bn_v, bn_b, bn_w, eps)
        conv = tf.nn.relu6(conv, name='relu6')

        return conv

def invres_oct_dl(inputs,
                weights,
                strides=1,
                is_train=True,
                name=None,
                padding='SAME',
                invres_idx=None):
    in_ch = (inputs.get_shape().as_list())[-1]
    with tf.variable_scope(name):
        conv = invres_pw_oct_dl(inputs, weights, strides=1, name='pw', padding=padding, invres_idx=invres_idx)
        conv = invres_dw_oct_dl(conv, weights, strides=strides, name='dw', padding=padding, invres_idx=invres_idx)
        conv = invres_pwl_oct_dl(conv, weights, strides=1, name='pw_linear', padding=padding, invres_idx=invres_idx)
    out_ch = (conv.get_shape().as_list())[-1]
    shortcut = ((strides==1) and (in_ch == out_ch))
    if shortcut:
        return conv + inputs
    else:
        return conv

def invres_pw_oct_dl(inputs,
                    weights,
                    strides=1,
                    name='pw',
                    padding='SAME',
                    invres_idx=None):
    eps = 1e-5
    prefix = 'module/features/' + str(invres_idx) + '/conv/'
    conv_w = weights[prefix + '0/weight']
    conv_w = conv_w.transpose((2,3,1,0))
    bn_w = weights[prefix + '1/weight']
    bn_b = weights[prefix + '1/bias']
    bn_m = weights[prefix + '1/running_mean']
    bn_v = weights[prefix + '1/running_var']
    with tf.variable_scope(name):
        #$ conv_w = get_variable_oct_dl(conv_w, 'conv_w')
        #$ bn_w = get_variable_oct_dl(bn_w, 'bn_w')
        #$ bn_b = get_variable_oct_dl(bn_b, 'bn_b')
        #$ bn_m = get_variable_oct_dl(bn_m, 'bn_m')
        #$ bn_v = get_variable_oct_dl(bn_v, 'bn_v')

        conv = tf.nn.conv2d(inputs, conv_w, strides=[1, strides, strides, 1], name='conv', padding=padding)
        conv = tf.nn.batch_normalization(conv, bn_m, bn_v, bn_b, bn_w, eps)
        conv = tf.nn.relu6(conv, name='relu6')

        return conv

def invres_dw_oct_dl(inputs,
                    weights,
                    strides=1,
                    name='dw',
                    padding='SAME',
                    invres_idx=None):
    eps = 1e-5
    prefix = 'module/features/' + str(invres_idx) + '/conv/'
    conv_w = weights[prefix + '3/weight']
    conv_w = conv_w.transpose((2,3,0,1))
    bn_w = weights[prefix + '4/weight']
    bn_b = weights[prefix + '4/bias']
    bn_m = weights[prefix + '4/running_mean']
    bn_v = weights[prefix + '4/running_var']
    with tf.variable_scope(name):
        #$ conv_w = get_variable_oct_dl(conv_w, 'conv_w')
        #$ bn_w = get_variable_oct_dl(bn_w, 'bn_w')
        #$ bn_b = get_variable_oct_dl(bn_b, 'bn_b')
        #$ bn_m = get_variable_oct_dl(bn_m, 'bn_m')
        #$ bn_v = get_variable_oct_dl(bn_v, 'bn_v')

        conv = tf.nn.conv2d(inputs, conv_w, strides=[1, strides, strides, 1], name='conv', padding=padding)
        conv = tf.nn.batch_normalization(conv, bn_m, bn_v, bn_b, bn_w, eps)
        conv = tf.nn.relu6(conv, name='relu6')

        return conv

def invres_pwl_oct_dl(inputs,
                    weights,
                    strides=1,
                    name='pw_linear',
                    padding='SAME',
                    invres_idx=None):
    eps = 1e-5
    prefix = 'module/features/' + str(invres_idx) + '/conv/'
    conv_w = weights[prefix + '6/weight']
    conv_w = conv_w.transpose((2,3,1,0))
    bn_w = weights[prefix + '7/weight']
    bn_b = weights[prefix + '7/bias']
    bn_m = weights[prefix + '7/running_mean']
    bn_v = weights[prefix + '7/running_var']
    with tf.variable_scope(name):
        #$ conv_w = get_variable_oct_dl(conv_w, 'conv_w')
        #$ bn_w = get_variable_oct_dl(bn_w, 'bn_w')
        #$ bn_b = get_variable_oct_dl(bn_b, 'bn_b')
        #$ bn_m = get_variable_oct_dl(bn_m, 'bn_m')
        #$ bn_v = get_variable_oct_dl(bn_v, 'bn_v')

        conv = tf.nn.conv2d(inputs, conv_w, strides=[1, strides, strides, 1], name='conv', padding=padding)
        conv = tf.nn.batch_normalization(conv, bn_m, bn_v, bn_b, bn_w, eps)

        return conv


def conv_1x1(input, w, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, w, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

def conv_1x1_expt(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d_expt(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)


def pwise_block(input, w, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, w, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, train=is_train, name='bn')
        out=relu(out)
        return out

"""
def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv

"""
def dwise_conv(input, w, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
                    padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def res_block(input, w, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim = round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, w[0], bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, w[1], strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, w[2], output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim = int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins = conv_1x1_expt(input, output_dim, name='ex_dim')
                net = ins + net
            else:
                net = input + net

        return net


def res_block_expt(input, w, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        """
        # pw
        bottleneck_dim = round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, w, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        """
        # dw
        net = dwise_conv(input, w[0], strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, w[1], output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim = int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins = conv_1x1_expt(input, output_dim, name='ex_dim')
                net = ins + net
            else:
                net = input + net

        return net



def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))

        pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1,stride, stride,1]

        conv=tf.nn.separable_conv2d(input,dwise_filter,pwise_filter,strides,padding=pad, name=name)
        if bias:
            biases = tf.get_variable('bias', [output_dim],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net
