import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_preprocessing_dataset1(data_loc, filename, results_loc):
    # Returns a usable data frame given the file name
    FEATURES = ['X1', 'X2']
    data_path = os.path.join(data_loc, filename)
    data = pd.read_csv(data_path, header=None, sep='\s+', names=FEATURES)
    n_observations = len(data)
    K = 2

    label = np.empty(shape=(0, 1))
    for i in range(K):
        label = np.append(label, np.ones((n_observations // K, 1), dtype=int) + i, axis=0)

    label = pd.DataFrame(label, columns=['label'], dtype=int)
    obs_nr = pd.DataFrame(np.arange(1, n_observations + 1), dtype=int, columns=['obs_i'])
    data = pd.concat([obs_nr, data, label], axis=1, join='inner')
    data = data.sample(frac=1)  # Shuffle rows

    plot_file_name = filename + '_raw_data_scatter.pdf'
    plt.scatter(data['X1'], data['X2'], s=1)
    plt.savefig(os.path.join(results_loc, plot_file_name))
    # plt.show()
    plt.close()

    return data, FEATURES, len(FEATURES)


def data_preprocessing_dataset2(data_loc, filename, results_loc):
    # Returns a usable data frame given the file name
    FEATURES = ['X1', 'X2']
    data_path = os.path.join(data_loc, filename)
    data = pd.read_csv(data_path, header=None, sep='\s+', names=FEATURES)
    n_observations = len(data)
    K = 3

    label = np.empty(shape=(0, 1))
    for i in range(K):
        label = np.append(label, np.ones((n_observations // K, 1), dtype=int) + i, axis=0)

    label = pd.DataFrame(label, columns=['label'], dtype=int)
    obs_nr = pd.DataFrame(np.arange(1, n_observations + 1), dtype=int, columns=['obs_i'])
    data = pd.concat([obs_nr, data, label], axis=1, join='inner')
    data = data.sample(frac=1)  # Shuffle rows

    plot_file_name = filename + '_raw_data_scatter.pdf'
    plt.scatter(data['X1'], data['X2'], s=1)
    plt.savefig(os.path.join(results_loc, plot_file_name))
    # plt.show()
    plt.close()

    return data, FEATURES, len(FEATURES)


def data_preprocessing_dataset3(data_loc, filename, label_file, results_loc):
    # Returns a usable data frame given the file name
    FEATURES = ['X1', 'X2']
    data_path = os.path.join(data_loc, filename)
    data = pd.read_csv(data_path, header=None, sep='\s+', names=FEATURES)
    n_observations = len(data)
    K = 2

    # label = np.empty(shape=(0, 1))
    # for i in range(K):
    #     label = np.append(label, np.ones((n_observations // K, 1), dtype=int) + i, axis=0)
    # label = pd.DataFrame(label, columns=['label'], dtype=int)
    label_path = os.path.join(data_loc, label_file)
    label = pd.read_csv(label_path, header=None, sep='\s+', names=['label'])

    obs_nr = pd.DataFrame(np.arange(1, n_observations + 1), dtype=int, columns=['obs_i'])
    data = pd.concat([obs_nr, data, label], axis=1, join='inner')
    data = data.sample(frac=1)  # Shuffle rows

    plot_file_name = filename + '_raw_data_scatter.pdf'
    plt.scatter(data['X1'], data['X2'], s=1)
    plt.savefig(os.path.join(results_loc, plot_file_name))
    # plt.show()
    plt.close()

    return data, FEATURES, len(FEATURES)
