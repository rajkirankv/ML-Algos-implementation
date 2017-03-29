import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data_path = '/Users/stan/PycharmProjects/CS274a/HW6/data'
# results_path = '/Users/stan/PycharmProjects/CS274a/HW6/results'
# iterations = 1  # This is the 'r' mentioned in the question i.e the number of times each calculation is repeated
# CONVERGENCE_CUTOFF = 1e-3  # This is the 'epsilon' smaller than which the points are considered overlapping
# K = 3
FEATURES = ['X1', 'X2']

logger = logging.Logger(__name__)
logger.setLevel('INFO')


def data_preprocesssing_dataset1(data_loc, filename, K, results_loc, labels=None):
    # Returns a usable data frame given the file name
    data_path = os.path.join(data_loc, filename)
    data = pd.read_csv(data_path, header=None, sep='\s+', names=FEATURES)
    n_observations = len(data)

    label = np.empty(shape=(0, 1))
    for i in range(K):
        label = np.append(label, np.ones((n_observations // K, 1), dtype=int) + i, axis=0)

    label = pd.DataFrame(label, columns=['label'], dtype=int)
    obs_nr = pd.DataFrame(np.arange(1, n_observations + 1), dtype=int, columns=['obs_i'])
    data = pd.concat([obs_nr, data, label], axis=1, join='inner')
    data = data.sample(frac=1)  # Shuffle rows

    plt.scatter(data['X1'], data['X2'], s=1)  # This plot will be saved and closed after plotting the results in the end
    results_file = os.path.join(results_loc, filename + '_Kmeans_rawdata_scatter.pdf')
    plt.savefig(results_file)
    plt.close()

    return data, n_observations


def data_preprocesssing_dataset2(data_loc, filename, K, results_loc, labels=None):
    # Returns a usable data frame given the file name
    data_path = os.path.join(data_loc, filename)
    data = pd.read_csv(data_path, header=None, sep='\s+', names=FEATURES)
    n_observations = len(data)

    label = np.empty(shape=(0, 1))
    for i in range(K):
        label = np.append(label, np.ones((n_observations // K, 1), dtype=int) + i, axis=0)

    label = pd.DataFrame(label, columns=['label'], dtype=int)
    obs_nr = pd.DataFrame(np.arange(1, n_observations + 1), dtype=int, columns=['obs_i'])
    data = pd.concat([obs_nr, data, label], axis=1, join='inner')
    data = data.sample(frac=1)  # Shuffle rows

    plt.scatter(data['X1'], data['X2'], s=1)  # This plot will be saved and closed after plotting the results in the end
    results_file = os.path.join(results_loc, filename + 'Kmeans_rawdata_scatter.pdf')
    plt.savefig(results_file)
    plt.close()

    return data, n_observations


def data_preprocesssing_dataset3(data_loc, filename, K, label_file, results_loc):
    # Returns a usable data frame given the file name
    data_path = os.path.join(data_loc, filename)
    data = pd.read_csv(data_path, header=None, sep='\s+', names=FEATURES)
    n_observations = len(data)

    label_path = os.path.join(data_loc, label_file)
    label = pd.read_csv(label_path, header=None, sep='\s+', names=['label'])

    obs_nr = pd.DataFrame(np.arange(1, n_observations + 1), dtype=int, columns=['obs_i'])
    data = pd.concat([obs_nr, data, label], axis=1, join='inner')
    data = data.sample(frac=1)  # Shuffle rows

    plt.scatter(data['X1'], data['X2'], s=1)  # This plot will be saved and closed after plotting the results in the end
    results_file = os.path.join(results_loc, filename + 'Kmeans_rawdata_scatter.pdf')
    plt.savefig(results_file)
    plt.close()

    return data, n_observations
