#!/usr/bin/env python

"""bpsk_generator.py: Provides a function that generates a BPSK dataset and saves it."""

import numpy as np
import csv

def bpsk_data(size=10000, energy=1, var=1, prob_one=0.5, filename=None):
    """ Generates binary signals randomly, and adds AWGN to simulate transmission through a noisy channel.
    Returns the noisy data and the original data to use in a machine learning algorithm. If filename is set, will save
    in a csv file as well as return.
    :param size: number of signals
    :param energy: energy of each signal.
    :param var: noise variance. If scalar, assumes equal variance. A tuple allows different variances for each signal.
    :param one_hot: True returns labels as one-hot vectors, False returns amplitude of the sent signals
    :param prob_one: probability of sending a 1
    :param filename: name of file to save data in
    :return: labels and received signals as separate numpy arrays
    """

    # enumerate amplitudes and probability of signals
    amplitudes = (-np.sqrt(energy), np.sqrt(energy))
    probabilities = (1-prob_one, prob_one)

    # generate sent signals
    sent = np.random.choice(a=amplitudes, p=probabilities, size=size)

    # add AWGN
    if type(var) == int or type(var) == float:
        received = sent + np.random.normal(loc=0.0, scale=np.sqrt(var), size=size)
    else:
        received = sent.copy()
        for i in range(len(received)):
            if received[i] == amplitudes[0]:
                received[i] += np.random.normal(loc=0.0, scale=np.sqrt(var[0]))
            else:
                received[i] += np.random.normal(loc=0.0, scale=np.sqrt(var[1]))

    # save data
    if filename is not None:
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            for i in range(len(sent)):
                writer.writerow([sent[i], received[i]])

    # return data as tuple of numpy arrays
    return sent.reshape((size, 1)), received.reshape((size, 1))

if __name__ == '__main__':
    sent, received = bpsk_data(size=10, filename='test_data.csv')
    print(sent)
    print(received)