#!/usr/bin/env python

"""model_builders.py: Group of helper functions for building neural network models"""

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import label_binarize
import os, contextlib
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    """ Generate weight variable using a standard normal distribution to initialise"""
    initial = tf.random_normal(shape, stddev=1.0, mean=0.0)
    return tf.Variable(initial)


def bias_variable(shape):
    """ Generate bias variable, initialising to 1.0"""
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """ Add tensorboard summaries to a variable
    :param var: tensorflow variable
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var-mean))))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('num', tf.count_nonzero(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, wd=0.0, losses_collection='losses'):
    """ Generate a neural network layer
    :param input_tensor: input placeholder or constant
    :param input_dim: dimensions of input
    :param output_dim: dimensions of output
    :param layer_name: name of layer (for scoping)
    :param act: activation function, defaults to rectified linear unit
    :param wd: weight decay scaling, defaults to 0 (no weight decay)
    :param losses_collection: name of the losses collection (for weight decays to be added to)
    :return: tensorflow object representing a node
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('pre_activations'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            variable_summaries(preactivate)
        if wd > 0:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd)
            tf.add_to_collection(losses_collection, weight_decay)
        with tf.name_scope('activations'):
            activations = act(preactivate, name='activation')
            variable_summaries(activations)
        return activations

def one_hot_encoder(y, positions=None):
    """ Takes a numpy array of values and encodes it to an array of one-hot vectors.
    :param y: values to encode.
    :param positions: list of all unique values in y, ordered by desired encoding. If None, this is done by default
                        sorting order.
    :return: one-hot encoded values

    """
    if positions == None:
        positions = np.unique(y)

    if len(np.unique(y)) == 2:
        return np.array([(1, 0) if i == positions[0] else (0, 1) for i in y])
    else:
        return label_binarize(y, positions)


def get_mnist_data(data_dir):
    # This function prints some annoying logging information, so the following code redirects it.
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            return input_data.read_data_sets(data_dir, one_hot=True)
