#!/usr/bin/env python

"""plot_conv_results.py: Plot results of mnist convolutional network tests."""

from helper_scripts.mnist_read_log import plot_results
import matplotlib.pyplot as plt

# Produce cross entropy and accuracy plots for mlp tests. Requires the training data for each of the models.

files = [r"""../mnist_conv_models/conv_nlayers=1_nfeatures=32/log/validation"""
         ]

scalar_names = ['accuracy_1', 'cross_entropy_1']
ylabels = ['Validation Accuracy', 'Cross Entropy (Validation Set)']
legend = [r'$n_{layers}=1, n_{units}=32$']

plot_results(files, scalar_names, ylabels, legend, 'Convolutional Models')


plt.show()