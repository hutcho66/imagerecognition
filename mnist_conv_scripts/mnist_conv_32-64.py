#!/usr/bin/env python

"""mnist_conv_32-64.py: Applies a 2 layer convolutional classifier to the MNIST digit dataset."""

import os

import tensorflow as tf

from helper_scripts import model_builders

learning_rate = 1e-3
weight_decay = 0.0001
dropout = 0.9
training_steps = 500000
batch_size = 1000

def train(weight_decay=0.0, keep_prob=1.0):

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

    # Define reshaping (for image example display)
    #with tf.name_scope('input_reshape'):
    #    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    #    tf.summary.image('input', image_shaped_input, 10)

    # Layer with 32 features, 5x5 patch and unit stride.
    with tf.name_scope('conv_layer1'):
        W_1 = model_builders.weight_variable([5, 5, 1, 32])
        b_1 = model_builders.bias_variable([32])
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])  # Reshape input image
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_reshape, W_1, strides=[1, 1, 1, 1], padding='SAME') + b_1)

    # Add max pooling from a 2x2 grid
    with tf.name_scope('pooling1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer with 64 features, 5x5 patch and unit stride.
    with tf.name_scope('conv_layer2'):
        W_2 = model_builders.weight_variable([5, 5, 32, 64])
        b_2 = model_builders.bias_variable([64])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_2, strides=[1, 1, 1, 1], padding='SAME') + b_2)

    # Add max pooling from a 2x2 grid
    with tf.name_scope('pooling2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Add a fully connected layer
    with tf.name_scope('fully_connected'):
        # Flatten layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3136])
        h_fc = model_builders.nn_layer(h_pool2_flat, 3136, 1024, 'fc_1', wd=weight_decay, act=tf.nn.relu)


    # Add dropout
    with tf.name_scope('dropout'):
        h_fc_dropped = tf.nn.dropout(h_fc, k)

    # Define the intermediate layer. For softmax outputs, we define an pre-activation layer to avoid numerical
    # instability issues.
    y = model_builders.nn_layer(h_fc_dropped, 1024, 10, 'intermediate', wd=weight_decay, act=tf.identity)

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
        # Every 100 steps, evaluate accuracy on validation set
        if i % 1000 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(training_data=False))
            validation_writer.add_summary(summary, i)
            # Print every 100 steps
            if i % 1000 == 0:
                print('Accuracy at step %s: %s' % (i, acc))
        else:
            # Every 1000 steps, record training metadata. Otherwise just do a training step and record results
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
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(training_data=True))
                train_writer.add_summary(summary, i)

    # Print final accuracys and save model
    # Entire training set is too big, so break into 11 smaller bunches of 5000 examples:
    train_acc = []
    for i in range(11):
        xs, ys = mnist.train.next_batch(batch_size=5000)
        train_acc.append(sess.run(accuracy, feed_dict={x: xs, t: ys, k: 1.0}))
    train_acc = sum(train_acc)/11  # average accuracy for the whole set
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

    print("Training with learning rate of %s, weight decay strength %s and keep probability %s." %
          (learning_rate, weight_decay, dropout))
    print("Using 2 convolutional layers with 32/64 features.")

    # Generate directory structure
    model_dir = "../mnist_conv_models/conv_nlayers=2_nfeatures=32-64"
    model_dir = os.path.abspath(model_dir)
    log_dir = os.path.join(model_dir, 'log')

    # Run training loop
    train(weight_decay=weight_decay, keep_prob=dropout)

    # Print information about TensorBoard.
    if tensorboard_info:
        print("\nSummaries stored in:", os.path.abspath(log_dir))
        print("To access TensorBoard, run the following command in a terminal: tensorboard --logdir=",
              os.path.abspath(log_dir), sep="")
        print(
            "The terminal program will then start a web application on localhost:6006 or another port if this one is",
            "already being used.")
        print("You must close the terminal program (CTRL-C) before running this program again.")