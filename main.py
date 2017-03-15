import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '/Users/stan/PycharmProjects/CS274a/HW6/data'
RESULTS_PATH = '/Users/stan/PycharmProjects/CS274a/HW6/results'
ITERATIONS = 1  # This is the 'r' mentioned in the question i.e the number of times each calculation is repeated
CONVERGENCE_CUTOFF = 1e-5  # This is the 'epsilon' smaller than which the points are considered overlapping
K = 2
FEATURES = ['X1', 'X2']


def data_preprocesssing(filename, labels=None):
    # Returns a usable data frame given the file name

    data_path = os.path.join(DATA_PATH, filename)
    data = pd.read_csv(data_path, header=None, sep='\s+', names=FEATURES)
    n_observations = len(data)
    temp1 = np.ones(n_observations // 2)
    temp2 = np.ones(n_observations // 2) + 1
    label = np.append(temp1, temp2)
    del temp1, temp2
    label = pd.DataFrame(label, columns=['label'], dtype=int)
    obs_nr = pd.DataFrame(np.arange(1, n_observations + 1), dtype=int, columns=['obs_i'])
    data = pd.concat([obs_nr, data, label], axis=1, join='inner')
    data = data.sample(frac=1)  # Shuffle rows

    plt.scatter(data['X1'], data['X2'], s=1)
    plt.savefig(os.path.join(RESULTS_PATH, 'scatter.pdf'))

    return data, n_observations


def get_euclidean_distance(data, center=None):
    # scipy.spatial.distance.euclidean is another option but the wiki is down
    return np.linalg.norm(data - center, axis=0)


def get_sse(data, center=None):
    if center is None:
        center = np.mean(data)
    distances = get_euclidean_distance(data, center)
    sse = np.sum(distances ** 2)
    return sse


def assign_clusters(data, means, old_cluster_labels, new_cluster_labels):
    # Number of clusters must be equal to the number of components proposed
    centers = means[-1]
    if len(centers) != K:
        raise ValueError

    # old_distance = np.inf
    # data['old_clusters'] = np.inf
    list_of_clusters = list()

    for mean in centers:
        data[old_cluster_labels] = data[new_cluster_labels]
        data['old_distance'] = data['new_distance']

        # For each observation, get the euclidean distance from the mean of each cluster
        new_distance = np.linalg.norm(data[FEATURES] - mean, axis=1)  # New distances wrt the new mean
        cluster_identity = new_distance < data['old_distance']
        # This is an N sized boolean array that returns those observations whose distance from the new mean
        # is less that that from the previous mean

        data.loc[cluster_identity, new_cluster_labels] = mean
        data.loc[cluster_identity, 'new_distance'] = new_distance[cluster_identity]
        # list_of_clusters.append(list(data.loc[cluster_identity, 'obs_i'].astype(int)))

    new_means = np.empty(shape=(1, len(FEATURES)))
    for mean in centers:
        obs_index = list(data[new_cluster_labels] == mean) # For each K-mean, select observations that are closest to that mean
        new_mean = np.mean(data.loc[obs_index, FEATURES], axis=0)
        new_mean = np.array(new_mean)
        new_mean = new_mean.reshape((1, len(FEATURES)))
        new_means = np.vstack((new_means, new_mean))

    means.append(new_means[1:])
    return list_of_clusters  # This should be a list of length K


def update_means(data, list_of_clusters, means):
    new_means = np.empty(shape=(1, len(FEATURES)))
    K = 0
    for cluster in list_of_clusters:
        if len(cluster) == 0:  # The case when no points changed changed to this cluster. Keep last iteration's mean
            mean = means[-1][K]
        else:
            mean = np.mean(data.loc[cluster, FEATURES], axis=0)
        mean = np.array(mean)
        mean = mean.reshape((1, len(FEATURES)))
        new_means = np.vstack((new_means, mean))
        K += 1
    means.append(new_means[1:])


def get_convergence_status(means):
    equality = np.isclose(means[-1], means[-2], atol=CONVERGENCE_CUTOFF)
    # Compare each feature(coordinate) of each of the K means to see if they are close enough
    return equality.all()


def get_K_means(data, seed_means, K=K):
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
    # At this point, data has these columns:
    # [obs_i, X1, X2, label, X1_old_cluster, X2_old_cluster, old_distance, X1_new_cluster, X2_new_cluster, new_distance]
    means = dict()
    means.append(np.array(seed_means[FEATURES]))
    while not convergence:

        # This is a list of subsets of observations such that each subset belongs to a distinct cluster
        assign_clusters(data, means, old_cluster_labels, new_cluster_labels)
        # means[-1] means using the most recently updated means
        update_means(data, clusters_list, means)

        convergence = get_convergence_status(means)
        print('Most recent mean: ')
        print(means[-1])
        print(np.linalg.norm(means[-1] - means[-2], axis=0))

    return means


def main(r=ITERATIONS):
    # dataset 1
    K = 2
    d = 2
    data, n_obs = data_preprocesssing('dataset1.txt')

    # Selecting from r random initializations
    sse_min = np.inf
    for trail in range(r):
        seed_means = data.sample(n=K)
        k_means = get_K_means(data, seed_means, K=K)
        sse = get_sse(data, center=k_means)
        if sse < sse_min:
            better_means = k_means
            sse_min = sse

    print(better_means)


main()
