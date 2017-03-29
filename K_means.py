import json
import logging
import os
import tkinter
import tkinter.filedialog as filedialog

import matplotlib.pyplot as plt
import numpy as np

from KMeans_data_preprocessing import \
    data_preprocesssing_dataset1, data_preprocesssing_dataset2, data_preprocesssing_dataset3
from K_means_utils import get_K_means

CONVERGENCE_CUTOFF = 1e-3  # This is the 'epsilon' smaller than which the points are considered overlapping
FEATURES = ['X1', 'X2']

logger = logging.Logger(__name__)
logger.setLevel('INFO')


def run_kmeans(data_path, file_name, K, results_path, results_file, iter_nr=1):
    if file_name == 'dataset1.txt':
        data, n_obs = data_preprocesssing_dataset1(data_path, file_name, K, results_path)
    elif file_name == 'dataset2.txt':
        data, n_obs = data_preprocesssing_dataset2(data_path, file_name, K, results_path)
    elif file_name == 'dataset3.txt':
        data, n_obs = data_preprocesssing_dataset3(data_path, file_name, K, 'labelset3.txt', results_path)
    else:
        print('Please provide a valid file')
        return

    seed_means = data.sample(n=K)
    k_means, sse = get_K_means(data, seed_means, K, file_name, results_path)
    print('Trail {}: SSE: {}'.format(iter_nr + 1, sse))
    return data, k_means, sse


def get_user_input():
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    currdir = os.getcwd()

    print('Please select a directory that contains the data')
    tempdir = filedialog.askdirectory(parent=root, initialdir=currdir,
                                      title='Please select a directory that contains the data')
    if len(tempdir) > 0:
        print('Looking for data in ' + tempdir)
        data_path = tempdir
    else:
        print('No directory selected. Exiting..')
        exit()

    print('Please select a directory to save the results')
    tempdir = filedialog.askdirectory(parent=root, initialdir=currdir,
                                      title='Please select a directory to save the results')
    if len(tempdir) > 0:
        print('Results will be saved in ' + tempdir)
        results_path = tempdir
    else:
        print('No directory selected. Results will be saved in ' + currdir)

    return data_path, results_path


# if __name__ == '__main__':
def kmeans(file_names):
    interactive = False

    if interactive:
        data_path, results_path = get_user_input()
    else:
        data_path = os.path.join(os.getcwd(), 'data')
        results_path = os.path.join(os.getcwd(), 'results')

    print('Enter the number of iterations for K means: ', end='')
    s = input()
    iterations = int(s)
    # file_names = [('dataset1.txt', 2), ('dataset2.txt', 3), ('dataset3.txt', 2)]

    sse_min = np.inf
    for file_name in file_names:
        print(file_name[0])
        for r in range(iterations):
            data, k_means, sse = run_kmeans(data_path=data_path, file_name=file_name[0], K=file_name[1],
                                            results_path=results_path,
                                            results_file='results' + file_name[0], iter_nr=r)
            if sse < sse_min:
                better_means = k_means
                better_labelled_data = data
                sse_min = sse

        final_result = 'final K means:\n {}, \n Final SSE:\n {}'.format(better_means, sse_min)
        print(file_name[0])
        print(final_result)
        results_file = file_name[0][:-4] + '_results_Kmeans'
        result_file_name = os.path.join(results_path, results_file)

        with open(result_file_name, 'at') as f:
            f.write(json.dumps(final_result))
        print('Final results stored at ' + result_file_name + ' in json string format')

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        markers = ['o', 'p', '+', '.', '*']
        K = file_name[1]

        for cluster in range(1, K + 1):
            cluster_identity = (better_labelled_data['label'] == cluster)
            plt.scatter(better_labelled_data.loc[cluster_identity, 'X1'],
                        better_labelled_data.loc[cluster_identity, 'X2'],
                        marker=markers[cluster - 1], color=colors[cluster - 1], s=5)
            plt.scatter(better_means[cluster - 1, 0], better_means[cluster - 1, 1],
                        marker='D', color=colors[cluster - 1], s=50)

        results_file = os.path.join(results_path, file_name[0] + '_final_scatter.pdf')
        plt.savefig(results_file)
        plt.close()
        plt.close('all')
