from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import batchnorm, prelu_tf, pixelShuffler, conv2


# The dense layer
def denseConvlayer(layer_inputs, bottleneck_scale, growth_rate, is_training):
    # Build the bottleneck operation
    net = layer_inputs
    net_temp = tf.identity(net)
    net = batchnorm(net, is_training)
    net = prelu_tf(net, name='Prelu_1')
    net = conv2(net, kernel=1, output_channel=bottleneck_scale * growth_rate, stride=1, use_bias=False, scope='conv1x1')
    net = batchnorm(net, is_training)
    net = prelu_tf(net, name='Prelu_2')
    net = conv2(net, kernel=3, output_channel=growth_rate, stride=1, use_bias=False, scope='conv3x3')

    # Concatenate the processed feature to the feature
    net = tf.concat([net_temp, net], axis=3)

    return net


# The transition layer
def transitionLayer(layer_inputs, output_channel, is_training):
    net = layer_inputs
    net = batchnorm(net, is_training)
    net = prelu_tf(net)
    net = conv2(net, 1, output_channel, stride=1, use_bias=False, scope='conv1x1')

    return net


# The dense block
def denseBlock(block_inputs, num_layers, bottleneck_scale, growth_rate, FLAGS):
    # Build each layer consecutively
    net = block_inputs
    for i in range(num_layers):
        with tf.compat.v1.variable_scope('dense_conv_layer%d' % (i + 1)):
            net = denseConvlayer(net, bottleneck_scale, growth_rate, FLAGS.is_training)

    return net


# Here we define the dense block version generator
def generatorDense(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # The main netowrk
    with tf.compat.v1.variable_scope('generator_unit', reuse=reuse):
        # The input stage
        with tf.compat.v1.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        # The dense block part
        # Define the denseblock configuration
        layer_per_block = 16
        bottleneck_scale = 4
        growth_rate = 12
        transition_output_channel = 128
        with tf.compat.v1.variable_scope('denseBlock_1'):
            net = denseBlock(net, layer_per_block, bottleneck_scale, growth_rate, FLAGS)

        with tf.compat.v1.variable_scope('transition_layer_1'):
            net = transitionLayer(net, transition_output_channel, FLAGS.is_training)

        with tf.compat.v1.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.compat.v1.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.compat.v1.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

        return net
