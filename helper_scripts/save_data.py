#!/usr/bin/env python

"""save_data.py: Saves data as csv for plotting. Requires full training data."""

from helper_scripts.mnist_read_log import get_scalar
import csv

file = r"""../mnist_conv_models/conv_nlayers=2_nfeatures=32-64/log/validation"""

data = get_scalar(file, ['accuracy_1', 'cross_entropy_1'])


with open('../plotting_data/conv_32-64.csv', 'w+') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')
    writer.writerow(('time', 'step', 'acc', 'entropy'))
    writer.writerows(zip(*data))