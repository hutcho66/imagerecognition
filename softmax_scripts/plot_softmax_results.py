#!/usr/bin/env python

"""plot_softmax_results.py: Plot results of mnist softmax tests."""

from helper_scripts.mnist_read_log import plot_results
import matplotlib.pyplot as plt

# Produce cross entropy and accuracy plots for softmax models.
# Requires the training data for each of the models.

files = [r"""../mnist_softmax_models\softmax_alpha=0.1_keepprob=0.9\log\validation"""
         ]

scalar_names = ['accuracy_1', 'cross_entropy_1']
ylabels = ['Validation Accuracy', 'Cross Entropy (Validation Set)']
legend = [r'$\alpha=0.1, keep\_prob=0.9$']

plot_results(files, scalar_names, ylabels, legend, 'Softmax Models')

plt.show()