#!/usr/bin/env python

"""mnist_evaluate_test.py: Evaluate a trained model on the test set. Doesn't require full training data, only model
                            checkpoints."""

import os

import tensorflow as tf

from helper_scripts import model_builders

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide warnings for non-optimised installation

data_dir = "../mnist_data"  # directory to store/access downloaded MNIST dataset
model_dir = r"""../mnist_mlp_models_regularised/mlp_nlayers=1_nunits=200/"""
model_meta_filename = "model.meta"

# Restore the model from the most recent save in the directory
sess = tf.Session()
saver = tf.train.import_meta_graph(model_dir+model_meta_filename)
saver.restore(sess, tf.train.latest_checkpoint(model_dir))

acc = sess.graph.get_tensor_by_name("accuracy/accuracy/Mean:0")
x = sess.graph.get_tensor_by_name("input/x-input:0")
t = sess.graph.get_tensor_by_name("input/t-input:0")
try:
    k = sess.graph.get_tensor_by_name("input/k:0")
    dropout = True
except KeyError:
    dropout = False

mnist = model_builders.get_mnist_data(data_dir)

if dropout:
    valid_acc = sess.run(acc, feed_dict={x: mnist.validation.images, t: mnist.validation.labels, k:1.0})
    test_acc = []
    for i in range(2):
        xs, ys = mnist.test.next_batch(batch_size=5000)
        test_acc.append(sess.run(acc, feed_dict={x: xs, t: ys, k: 1.0}))
    test_acc = sum(test_acc) / 2  # average accuracy for the whole set
else:
    valid_acc = sess.run(acc, feed_dict={x: mnist.validation.images, t: mnist.validation.labels})
    test_acc = []
    for i in range(2):
        xs, ys = mnist.test.next_batch(batch_size=5000)
        test_acc.append(sess.run(acc, feed_dict={x: xs, t: ys}))
    test_acc = sum(test_acc) / 2  # average accuracy for the whole set
print(model_dir)
print("Validation Accuracy: %s." % valid_acc)
print("Test Accuracy: %s." % test_acc)
print('\n')
sess.close()


