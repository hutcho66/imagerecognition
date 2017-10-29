#!/usr/bin/env python

"""mnist_mlp.py: Applies a hidden layer classifier to the MNIST digit dataset."""

import os

import tensorflow as tf

from helper_scripts import model_builders

# Add all models to be trained as tuples.
models = [(2,150), (2,100), (2,50)]
learning_rate = 1e-3
weight_decay = 0.0001
dropout = 0.9
training_steps = 200000
batch_size = 1000


def train(weight_decay=0.0, keep_prob=1.0, num_layers=1, num_units=100):

    mnist = model_builders.get_mnist_data(data_dir)

    sess = tf.Session()

    def feed_dict(training_data=True):
        """ Generates a feed dictionary.
        :param training_data: Boolean to flag which dataset to use.
        :return: feed dictionary

        """
        if training_data:
            xs, ys = mnist.train.next_batch(batch_size)
            kp = keep_prob
        else:
            xs, ys = mnist.validation.images, mnist.validation.labels
            kp = 1.0
        return {x: xs, t: ys, k: kp}

    # Define input tensors
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
        t = tf.placeholder(tf.float32, shape=[None, 10], name='t-input')
        k = tf.placeholder(tf.float32, name='k')

    # Define the hidden layers.
    layers = [x]
    for i in range(num_layers):
        layers.append(tf.nn.dropout(layers[-1], k))
        layers.append(model_builders.nn_layer(layers[-1], int(layers[-1].get_shape()[1]), num_units, 'hidden_%s' % (i+1),
                                              wd=weight_decay, act=tf.nn.relu))
    layers.append(tf.nn.dropout(layers[-1], k))

    # Define the intermediate layer. For softmax outputs, we define an pre-activation layer to avoid numerical
    # instability issues.
    y = model_builders.nn_layer(layers[-1], int(layers[-1].get_shape()[1]), 10, 'intermediate', wd=weight_decay, act=tf.identity)

    # Define the cross entropy operation
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y)
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.add_to_collection('losses', cross_entropy)

    # Combine losses
    with tf.name_scope('total_loss'):
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('total_loss', loss)

    # Define the training step
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Define the accuracy measurements
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summary data and save to log files
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(log_dir + '/validation', sess.graph)

    # Initialise global variables
    sess.run(tf.global_variables_initializer())

    # Perform training
    for i in range(training_steps):
        # Every 1000 steps, evaluate accuracy on validation set
        if i % 100 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(training_data=False))
            validation_writer.add_summary(summary, i)
            # Print every 1000 steps
            if i % 1000 == 0:
                print('Accuracy at step %s: %s' % (i, acc))
        else:
            # Every 1000 steps, record training metadata. Otherwise just do a training step and record results
            if i % 10000 == 9999:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(training_data=True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                #print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(training_data=True))
                train_writer.add_summary(summary, i)

    # Print final accuracys and save model
    train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, t: mnist.train.labels, k: 1.0})
    val_acc = sess.run(accuracy, feed_dict=feed_dict(training_data=False))
    print("Model trained. Validation accuracy is %s. Training accuracy is %s." % (val_acc, train_acc))

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(model_dir, 'model'))

    train_writer.close()
    validation_writer.close()

    tf.reset_default_graph()
    sess.close()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide warnings for non-optimised installation

    data_dir = "../mnist_data"  # directory to store/access downloaded MNIST dataset
    tensorboard_info = False

    for model in models:
        nlayers = model[0]
        nunits = model[1]
        print("Training with learning rate of %s, weight decay strength %s and keep probability %s." %
              (learning_rate, weight_decay, dropout))
        print("Using %s hidden layer/s and %s units per layer." % (nlayers, nunits))

        # Generate directory structure
        model_dir = "../mnist_mlp_models"
        model_dir = os.path.join(os.path.abspath(model_dir),
                                 'mlp_nlayers=%s_nunits=%s' % (nlayers, nunits))
        log_dir = os.path.join(model_dir, 'log')

        # Run training loop
        train(weight_decay=weight_decay, keep_prob=dropout, num_layers=nlayers, num_units=nunits)

        # Print information about TensorBoard.
        if tensorboard_info:
            print("\nSummaries stored in:", os.path.abspath(log_dir))
            print("To access TensorBoard, run the following command in a terminal: tensorboard --logdir=",
                  os.path.abspath(log_dir), sep="")
            print(
                "The terminal program will then start a web application on localhost:6006 or another port if this one is",
                "already being used.")
            print("You must close the terminal program (CTRL-C) before running this program again.")