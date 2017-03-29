import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DATA_PATH = '/Users/stan/PycharmProjects/CS274a/HW6/data'
# RESULTS_PATH = '/Users/stan/PycharmProjects/CS274a/HW6/results'
# ITERATIONS = 1  # This is the 'r' mentioned in the question i.e the number of times each calculation is repeated
CONVERGENCE_CUTOFF = 1e-5  # This is the 'epsilon' smaller than which the points are considered overlapping
FEATURES = ['X1', 'X2']

logger = logging.Logger(__name__)
logger.setLevel('DEBUG')


def get_sse(data, K, center=None):
    if center is None:
        center = np.mean(data)

    sse = 0
    for cluster_label in range(1, K + 1):
        cluster_identity = (data['new_cluster_label'] == cluster_label)
        sse += np.sum(np.linalg.norm(data.loc[cluster_identity, FEATURES] - center[cluster_label - 1, :], axis=0) ** 2)
    return sse


def assign_clusters(data, means, old_cluster_labels, new_cluster_labels):
    # Number of clusters must be equal to the number of components proposed
    centers = means[-1]
    # if len(centers) != K:
    #     raise ValueError

    # old_distance = np.inf
    # data['old_clusters'] = np.inf
    # list_of_clusters = list()
    cluster_label = 1
    for mean in centers:
        data[old_cluster_labels] = data[new_cluster_labels]
        data['old_distance'] = data['new_distance']

        # For each observation, get the euclidean distance from the mean of each cluster
        new_distance = np.linalg.norm(data[FEATURES] - mean, axis=1)  # New distances wrt the new mean
        cluster_identity = new_distance < data['old_distance']
        # This is an N sized boolean array that returns those observations whose distance from the new mean
        # is less that that from the previous mean
        if any(cluster_identity):
            data.loc[cluster_identity, new_cluster_labels] = mean
            data.loc[cluster_identity, 'new_distance'] = new_distance[cluster_identity]
            data.loc[cluster_identity, 'new_cluster_label'] = cluster_label
        cluster_label += 1


def update_means(data, means, K):
    new_means = np.empty(shape=(0, len(FEATURES)))
    for cluster_label in range(1, K + 1):
        cluster = (data['new_cluster_label'] == cluster_label)
        mean = np.mean(data.loc[cluster, FEATURES], axis=0)
        mean = np.array(mean)
        mean = mean.reshape((1, len(FEATURES)))
        new_means = np.vstack((new_means, mean))
    means.append(new_means)


def get_convergence_status(means):
    equality = np.isclose(means[-1], means[-2], atol=CONVERGENCE_CUTOFF)
    # Compare each feature(coordinate) of each of the K means to see if they are close enough
    return equality.all()


def get_K_means(data, seed_means, K, file_name, results_loc):
    # In the beginning, data has these columns: [obs_i, X1, X2, label]
    convergence = False

    # Create new columns for cluster features
    # The labels can be improved by using a multilevel index
    old_cluster_labels = list()
    new_cluster_labels = list()
    for feature in FEATURES:
        old_cluster_labels.append(feature + '_old_cluster')
        new_cluster_labels.append(feature + '_new_cluster')

    data = pd.concat([data, pd.DataFrame(columns=old_cluster_labels)])
    data['old_distance'] = np.inf
    data = pd.concat([data, pd.DataFrame(columns=new_cluster_labels)])
    data['new_distance'] = np.inf
    data = pd.concat([data, pd.DataFrame(columns=['new_cluster_label'])])
    # At this point, data has these columns:
    # [obs_i, X1, X2, label, X1_old_cluster, X2_old_cluster, old_distance, X1_new_cluster, X2_new_cluster, new_distance, new_cluster_label]

    means = list()
    # changed_distance = list()
    means.append(np.array(seed_means[FEATURES]))
    sse_per_obs = list()
    N = len(data.index)
    i = 0
    iterations = list()

    while not convergence:
        # This is a list of subsets of observations such that each subset belongs to a distinct cluster
        i += 1
        iterations.append(i)
        assign_clusters(data, means, old_cluster_labels, new_cluster_labels)
        update_means(data, means, K)
        sse = get_sse(data, K, center=means[-1])
        sse_per_obs.append(sse / N)
        convergence = get_convergence_status(means)
        # changed_distance.append(np.sum(np.linalg.norm(means[-1] - means[-2], axis=0)))

    # plt.scatter(x=iterations, y=changed_distance, s=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Total change in distance of K means')
    # results_file = os.path.join(RESULTS_PATH, file_name + '_moving_points.pdf')
    # plt.savefig(results_file)
    # plt.close()

    plt.scatter(x=iterations, y=sse_per_obs, s=2)
    plt.xlabel('Iterations')
    plt.ylabel('SSE per data point')
    results_file = os.path.join(results_loc, file_name + '_sse_convergence.pdf')
    plt.savefig(results_file)
    plt.close()

    return means[-1], sse_per_obs[-1]
