#!/usr/bin/env python

"""bpsk_softmax.py: Applies a softmax classifier to BPSK data."""

import os

import tensorflow as tf

from bpsk_scripts import bpsk_generator
from helper_scripts import model_builders

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide warnings for non-optimised installation

log_dir = "bpsk_logs"  # TensorBoard logs directory
training_steps = 1000  # Number of epochs
learning_rate = 0.5

# Set BPSK signal parameters
training_set_size = 100000
test_set_size = 10000
energy = 4.0
var = 1.0  # Set to a tuple for unequal variances
prob_one = 0.5


def get_data(train=True):
    if train:
        y, x = bpsk_generator.bpsk_data(size=training_set_size, energy=energy, var=var, prob_one=prob_one)
        t = model_builders.one_hot_encoder(y)
    else:
        y, x = bpsk_generator.bpsk_data(size=test_set_size, energy=energy, var=var, prob_one=prob_one5)
        t = model_builders.one_hot_encoder(y)
    return x, t

def train():
    def feed_dict(ys, xs):
        """ Generates a feed dictionary.
        :param ys: y values, must be one_hot encoded
        :param xs: x values, must be same length as y values
        :param k: keep probability, default to 1 (always keep)
        :return: feed dictionary

        """
        return {x: xs, t: ys}

    sess = tf.Session()

    # Create input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 1], name='x-input')
        t = tf.placeholder(tf.float32, shape=[None, 2], name='t-input')

    # Create a layer with identity output to find the pre-activations
    y = model_builders.nn_layer(x, 1, 2, 'layer', act=tf.identity)

    # Find the average cross entropy given the current weights
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add cross entropy to loss and combine with any weight decay losses
    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # Define the training step, using a gradient descent optimiser to minimise the loss
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Calculate the accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summary data and save to log files
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # Initialise global variables
    sess.run(tf.global_variables_initializer())

    # Get training and test data
    training_x, training_t = get_data(train=True)
    test_x, test_t = get_data(train=True)

    for i in range(training_steps):
        # Every 10 steps, evaluate accuracy on training set
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(test_t, test_x))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            # Every 100 steps, record training metadata. Otherwise just do a training step and record results
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(training_t, training_x),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(training_t, training_x))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


if __name__ == "__main__":
    # Remove data from previous runs
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    # Run training loop
    train()

    print("Summaries stored in:", os.path.abspath(log_dir))
    print("To access TensorBoard, run the following command in a terminal: tensorboard --logdir=",
          os.path.abspath(log_dir), sep="")
    print("The terminal program will then start a web application on localhost:6006 or another port if this one is",
          "already being used.")
    print("You must close the terminal program (CTRL-C) before running this program again.")
