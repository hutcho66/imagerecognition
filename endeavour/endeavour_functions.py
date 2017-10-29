#!/usr/bin/env python

"""endeavour_functions.py: Various functions to assist the endeavour application"""

from PIL import Image, ImageOps
import numpy as np
import logging
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide warnings for non-optimised installation

def process_image(filename, dest_filename=None, threshold=190):
    image = Image.open(filename)
    image = image.convert('L')  # convert to greyscale

    # Convert to B&W and invert
    image = np.array(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > threshold:
                image[i][j] = 0
            else:
                image[i][j] = 255

    # Resize to 20x20 using anti-aliasing filter and add 4 pixel border to make 28x28
    image = Image.fromarray(image)
    image = image.resize((20,20), Image.ANTIALIAS)
    image = ImageOps.expand(image, border=4, fill='black')

    if dest_filename is not None:
        try:
            image.save(dest_filename)  # save as image
        except Exception:
            logging.exception("Can't save image")

    return np.array(image)  # return as array

def classify_image(im_filename, model_dir, model_filename):
    # Load image
    image = Image.open(im_filename)
    image = np.array(image)
    image = image / image.max()

    # Restore the model from the most recent save in the directory
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_dir + model_filename)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    x = sess.graph.get_tensor_by_name("input/x-input:0")
    y = sess.graph.get_tensor_by_name("intermediate/activations/activation:0")
    try:
        k = sess.graph.get_tensor_by_name("input/k:0")
        dropout = True
    except KeyError:
        dropout = False

    # Run example through model
    if dropout:
        feed_dict = {x: image.reshape((1, 784)), k: 1.0}
    else:
        feed_dict = {x: image.reshape((1, 784))}
    classification = sess.run(y, feed_dict)[0]
    classification = np.exp(classification) / sum(np.exp(classification))

    sess.close()
    tf.reset_default_graph()

    return np.argmax(classification), classification
