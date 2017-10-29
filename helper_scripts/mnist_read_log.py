#!/usr/bin/env python

"""mnist_read_log.py: Provides functions relating to tensorboard logs."""

from tensorflow.tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def get_scalar(file, scalar_names):
    """
    Gets times, step numbers and values for scalars in the log given
    :return: times, step numbers and values as a tuple of tuples
    :param file: filename of the log
    :param scalar_names: names of the scalars
    """
    event_acc = EventAccumulator(file)
    event_acc.Reload()

    # Zip first scalar
    output = list(zip(*event_acc.Scalars(scalar_names[0])))

    for i in range(1, len(scalar_names)):
        scalar = list(zip(*event_acc.Scalars(scalar_names[i])))[2]
        output.append(scalar)

    return output


def plot_results(dirnames, scalarnames, ylabels, legend, window_title):
    data = []
    for dir in dirnames:
        data.append(get_scalar(dir, scalarnames))

    for i, scalar in enumerate(scalarnames):
        plt.figure()
        plt.rc('font', family='serif')
        for j in range(len(dirnames)):
            plt.plot(data[j][1], data[j][2+i])
        plt.xlabel(r'Step Number')
        plt.ylabel(ylabels[i])
        plt.legend(legend)
        fig = plt.gcf()
        fig.canvas.set_window_title(scalar + ', ' + window_title)
