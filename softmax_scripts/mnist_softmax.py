#!/usr/bin/env python

"""mnist_softmax.py: Applies a softmax classifier to the MNIST digit dataset."""

import os

import tensorflow as tf

from helper_scripts import model_builders

# Models will be trained using the learning rates in the following list:
learning_rates = [0.001, 0.01, 0.1, 1]
training_steps = 1000000
batch_size = 1000

def train():

    mnist = model_builders.get_mnist_data(data_dir)

    sess = tf.Session()

    def feed_dict(training_data=True):
        """ Generates a feed dictionary.
        :param training_data: Boolean to flag which dataset to use.
        :return: feed dictionary

        """
        if training_data:
            xs, ys = mnist.train.next_batch(batch_size)
        else:
            xs, ys = mnist.validation.images, mnist.validation.labels
        return {x: xs, t: ys}

    # Define input tensors
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
        t = tf.placeholder(tf.float32, shape=[None, 10], name='t-input')

    # Define reshaping (for image example display)
    #with tf.name_scope('input_reshape'):
    #    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    #    tf.summary.image('input', image_shaped_input, 10)

    # Define the intermediate layer. For softmax outputs, we define an pre-activation layer to avoid numerical
    # instability issues.
    y = model_builders.nn_layer(x, 784, 10, 'intermediate', act=tf.identity)

    # Define the cross entropy operation
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Define the training step
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

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
        if i % 1000 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(training_data=False))
            validation_writer.add_summary(summary, i)
            # Print every 100000 steps
            if i % 100000 == 0:
                print('Accuracy at step %s: %s' % (i, acc))
        else:
            # Every 1000 steps, record training accuracy and metadata. Otherwise just do a training step.
            if i % 1000 == 999:
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
                sess.run([train_step], feed_dict=feed_dict(training_data=True))

    # Print final accuracies and save model
    train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, t: mnist.train.labels})
    val_acc = sess.run(accuracy, feed_dict=feed_dict(training_data=False))
    print("Model trained. Validation accuracy is %s. Training accuracy is %s." % (val_acc, train_acc))

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(model_dir, 'model'), global_step=i)

    train_writer.close()
    validation_writer.close()

    tf.reset_default_graph()
    sess.close()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide warnings for non-optimised installation

    data_dir = "../mnist_data"  # directory to store/access downloaded MNIST dataset
    tensorboard_info = False

    for learning_rate in learning_rates:
        print("Training with learning rate of %s." % learning_rate)

        # Generate directory structure
        model_dir = "../mnist_softmax_models"
        model_dir = os.path.join(os.path.abspath(model_dir), 'softmax_alpha=%s' % learning_rate)
        log_dir = os.path.join(model_dir, 'log')

        # Run training loop
        train()

        # Print information about TensorBoard.
        if tensorboard_info:
            print("\nSummaries stored in:", os.path.abspath(log_dir))
            print("To access TensorBoard, run the following command in a terminal: tensorboard --logdir=",
                  os.path.abspath(log_dir), sep="")
            print("The terminal program will then start a web application on localhost:6006 or another port if this one is",
                  "already being used.")
            print("You must close the terminal program (CTRL-C) before running this program again.")