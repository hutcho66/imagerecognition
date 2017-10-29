#!/usr/bin/env python

"""plot_mlp_results.py: Plot results of mnist mlp tests."""

from helper_scripts.mnist_read_log import plot_results
import matplotlib.pyplot as plt

# Produce cross entropy and accuracy plots for mlp tests. Requires the training data for each of the models.

files = [r"""../mnist_mlp_models/mlp_nlayers=2_nunits=150/log/validation/""",
         r"""../mnist_mlp_models/mlp_nlayers=2_nunits=100/log/validation/""",
         r"""../mnist_mlp_models/mlp_nlayers=2_nunits=50/log/validation/""",
         ]

scalar_names = ['accuracy_1', 'cross_entropy_1']
ylabels = ['Validation Accuracy', 'Cross Entropy (Validation Set)']
legend = [r'$n_{layers}=2, n_{units}=150$', r'$n_{layers}=2, n_{units}=100$', r'$n_{layers}=2, n_{units}=50$']

plot_results(files, scalar_names, ylabels, legend, 'MLP Models')


plt.show()