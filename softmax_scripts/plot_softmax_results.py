#!/usr/bin/env python

"""plot_softmax_results.py: Plot results of mnist softmax tests."""

from helper_scripts.mnist_read_log import plot_results
import matplotlib.pyplot as plt

# Produce cross entropy and accuracy plots for no regularisation tests. Requires the training data for each of the models.

files = [r"""../mnist_softmax_models\softmax_alpha=0.001\log\validation""",
         r"""../mnist_softmax_models\softmax_alpha=0.01\log\validation""",
         r"""../mnist_softmax_models\softmax_alpha=0.1\log\validation""",
         r"""../mnist_softmax_models\softmax_alpha=1\log\validation"""
         ]

scalar_names = ['accuracy_1', 'cross_entropy_1']
ylabels = ['Validation Accuracy', 'Cross Entropy (Validation Set)']
legend = [r'$\alpha=0.001$', r'$\alpha=0.01$', r'$\alpha=0.1$', r'$\alpha=1$']

plot_results(files, scalar_names, ylabels, legend, 'No Normalisation')

# Produce cross entropy and accuracy plots for dropout tests. Requires the training data for each of the models.

files = [r"""../mnist_softmax_models\softmax_alpha=0.1_keepprob=0.5\log\validation""",
         r"""../mnist_softmax_models\softmax_alpha=0.1_keepprob=0.7\log\validation""",
         r"""../mnist_softmax_models\softmax_alpha=0.1_keepprob=0.9\log\validation"""
         ]

scalar_names = ['accuracy_1', 'cross_entropy_1']
ylabels = ['Validation Accuracy', 'Cross Entropy (Validation Set)']
legend = [r'$k=0.5$', r'$k=0.7$', r'$k=0.9$']

plot_results(files, scalar_names, ylabels, legend, 'Dropout')

# Produce cross entropy and accuracy plots for weight decay tests. Requires the training data for each of the models.
files = [r"""../mnist_softmax_models\softmax_alpha=0.1_wd=0.0001\log\validation""",
         r"""../mnist_softmax_models\softmax_alpha=0.1_wd=0.001\log\validation""",
         r"""../mnist_softmax_models\softmax_alpha=0.1_wd=0.01\log\validation""",
         r"""../mnist_softmax_models\softmax_alpha=0.1_wd=0.1\log\validation"""
         ]

scalar_names = ['accuracy_1', 'cross_entropy_1']
ylabels = ['Validation Accuracy', 'Cross Entropy (Validation Set)']
legend = [r'$\lambda=0.0001$', r'$\lambda=0.001$', r'$\lambda=0.01$', r'$\lambda=0.1$']

plot_results(files, scalar_names, ylabels, legend, 'Weight Decay')

plt.show()