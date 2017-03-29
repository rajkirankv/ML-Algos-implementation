import json
import os
import tkinter
import tkinter.filedialog as filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as P

from EM_Algorithm_utils import initialize_parameters, get_comp_distributions, get_mixture_model, \
    avoid_singularity
from EM_data_preprocessing import data_preprocessing_dataset1, data_preprocessing_dataset2, \
    data_preprocessing_dataset3
from plot_data_and_gaussians import plot_data_and_gaussians

# -----These import statements seem to depend on the idiosyncrasies of the the IDE and commandline.
# In case of errors, flip the commenting and uncommenting
# from EM_Algorithm.EM_Algorithm_utils import initialize_parameters, get_comp_distributions, get_mixture_model, \
#     avoid_singularity
# from EM_Algorithm.data_preprocessing import data_preprocessing_dataset1, data_preprocessing_dataset2, \
#     data_preprocessing_dataset3

# data_loc = '/Users/stan/PycharmProjects/CS274a/HW6/data'
# results_loc = '/Users/stan/PycharmProjects/CS274a/HW6/results'
# iterations = 1  # This is the 'r' mentioned in the question i.e the number of times each calculation is repeated
SINGULARITY_EPSILON = 1e-4
CONVERGENCE_EPSILON = 1e-2


def e_step(D, alphas, means, covars):
    probabilities_matrix = get_comp_distributions(D, means=means, covars=covars)
    probs_from_mixture_model = get_mixture_model(D, alphas=alphas, means=means,
                                                 covars=covars, probabilities_matrix=probabilities_matrix)
    W = np.multiply(probabilities_matrix, alphas.T)

    # --------DEBUG for 0/0s------
    # It is possible for a given set of parameters, the probability contributed by each component and hence the total
    # probabbility from the mixture model is zero for an observation. This results in a 0/0 situation for the weight.
    # This step takes care of the errors occuring due to this
    zeros = (probs_from_mixture_model == 0)
    # np.squeeze(zeros)
    if zeros.any():
        probs_from_mixture_model[zeros] = SINGULARITY_EPSILON
        W[zeros] = SINGULARITY_EPSILON / len(alphas)
    # --------DEBUG for 0/0s------

    W = np.divide(W, probs_from_mixture_model)

    N = D.shape[0]
    if int(np.sum(W)) != N:
        print('Warning: Sum of weights of K mixture components do not add up')

    return W, probabilities_matrix


def m_step(D, W):
    N = np.sum(W)  # This is the sum of N_ks
    N_k = np.sum(W, axis=0).T  # This is the K * 1 vector of N_ks
    N_k = np.expand_dims(N_k, axis=1)
    new_alphas = N_k / N  # K * 1 vector

    new_means = np.dot(W.T, D)  # K * d matrix
    new_means = np.divide(new_means, N_k)  # K * d matrix

    # d = D.shape[1] # get the dimension of observations
    # D_in_3D = np.dstack([D]*K) # Stack K copies of D matrix on top of each other
    # new_means.reshape(1, d, K) # reshape the matrix of means to facilitate calculation
    # centered_D = D_in_3D - new_means
    # new_covars = np.dot(centered_D.T, centered_D)

    # ------This part needs improvements-------#
    N = D.shape[0]
    K = W.shape[1]
    new_covars = list()
    for k in range(K):
        covar_k = 0
        for i in range(1, N):
            centered_obs = D[i, :] - new_means[k, :]  # this is 1 * d
            centered_obs = np.expand_dims(centered_obs, axis=1)
            covar_k_comp = np.dot(centered_obs, centered_obs.T)  # this is d * d
            covar_k_comp = W[i, k] * covar_k_comp
            covar_k += covar_k_comp
        avoid_singularity(covar_k, SINGULARITY_EPSILON)
        new_covars.append(covar_k)
    new_covars = np.dstack(tuple(new_covars)).T
    for k in range(K):
        new_covars[k] /= N_k[k]
    # ------This part needs improvements-------#

    return new_means, new_covars, new_alphas


def get_log_likelihood(probabilities_matrix, alphas):
    N = probabilities_matrix.shape[0]
    log_likelihood = np.dot(probabilities_matrix, alphas)  # This is N * 1
    log_likelihood = np.dot(np.ones(shape=(1, N)), log_likelihood)
    return np.asscalar(np.squeeze(log_likelihood))


def run_em(data_path, file_name, K, results_path, results_file, iter_nr=1, bic_call=False):
    # D: N * d matrix
    # means: K * d matrix
    # covars: d * d * K matrix
    # W: N * K matrix
    # probabilities_matrix: N(observations) * K(components) matrix

    if file_name == 'dataset1.txt':
        data, FEATURES, N_DIMS = data_preprocessing_dataset1(data_path, file_name, results_path)
    elif file_name == 'dataset2.txt':
        data, FEATURES, N_DIMS = data_preprocessing_dataset2(data_path, file_name, results_path)
    elif file_name == 'dataset3.txt':
        data, FEATURES, N_DIMS = data_preprocessing_dataset3(data_path, file_name, 'labelset3.txt', results_path)
    else:
        print('Please provide a valid file')
        return

    data = np.array(data[FEATURES])
    means, covars, alphas = initialize_parameters(data, dims=N_DIMS, K=K, type='random')

    # --Plot initial parameters
    if bic_call:
        # Skip plotting
        pass
    else:
        absolute_file_path = os.path.join(data_path, file_name)
        plot_data_and_gaussians(absolute_file_path, means, covars)
        # P.show()
        results_file_path = os.path.join(results_path, file_name[:-4] + '_' + str(iter_nr) + '_initial_pms.pdf')
        P.savefig(results_file_path)
        P.close()

    convergence = False
    log_likelihood = list()
    log_likelihood.append(-np.inf)
    # log_likelihood.append(get_log_likelihood(data, means_old, covars_old, alphas_old))

    current_log_likelihood = 0
    cumulative_log_likelihood = 0
    average_log_likelihood = 0
    iterations = list()
    i = 0
    while not convergence:
        i += 1
        iterations.append(i)
        alphas_old = alphas
        W, probabilities_matrix = e_step(data, alphas, means, covars)
        means, covars, alphas = m_step(data, W)
        current_log_likelihood = get_log_likelihood(probabilities_matrix, alphas_old)
        cumulative_log_likelihood += current_log_likelihood
        average_log_likelihood = cumulative_log_likelihood / i
        log_likelihood.append(current_log_likelihood)
        if not bic_call:
            print(log_likelihood[-1])
        if (log_likelihood[-1] - log_likelihood[-2]) < -average_log_likelihood:
            # raise ValueError('Log likelihood is decreasing')
            print('Warning: Log likelihood is decreasing')
        elif abs(log_likelihood[-1] - log_likelihood[-2]) < CONVERGENCE_EPSILON:
            convergence = True
        else:
            continue

    # PLOT
    x_axis = list(iterations)
    plt.plot(x_axis, log_likelihood[1:])
    plt.xlabel('Iteration')
    plt.ylabel('Log likelihood')
    result_file_name = 'EM_log_likelihood_' + file_name[:-4] + '_' + str(iter_nr) + '.pdf'
    result_file_name = os.path.join(results_path, result_file_name)
    plt.savefig(result_file_name)
    # plt.show()
    plt.close()

    # --Plot final parameters
    if bic_call:
        # Skip plotting
        pass
    else:
        absolute_file_path = os.path.join(data_path, file_name)
        plot_data_and_gaussians(absolute_file_path, means, covars)
        # P.show()
        results_file_path = os.path.join(results_path, file_name[:-4] + '_' + str(iter_nr) + '_final_pms.pdf')
        P.savefig(results_file_path)
        P.close()

    final_result = 'final weights:\n {}, \n means:\n {}, \n covars:\n {}, \n log_likelihood:\n' \
                   ' {}'.format(alphas, means, covars, log_likelihood[-1])
    if not bic_call:
        print(file_name + '\n' + final_result)
    result_file_name = os.path.join(results_path, results_file)
    with open(result_file_name, 'at') as f:
        f.write(json.dumps(final_result))
    if not bic_call:
        print('Final results stored at ' + result_file_name + ' in json string format')

    return log_likelihood[-1], data.shape[0]


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
def em(file_names, K=None, bic_call=False):
    interactive = False

    if interactive:
        data_path, results_path = get_user_input()
    else:
        data_path = os.path.join(os.getcwd(), 'data')
        results_path = os.path.join(os.getcwd(), 'results')

    if K is None:
        # Avoid asking for number of iterations while doing BIC
        print('Enter the number of iterations for EM: ', end='')
        s = input()
        ITERATIONS = int(s)
    else:
        ITERATIONS = 1

    log_likelihood_for_bic = 0

    for file_name in file_names:
        for r in range(ITERATIONS):
            cur_log_likelihood_for_bic, N = run_em(data_path=data_path, file_name=file_name[0],
                                                   K=file_name[1] if K is None else K,
                                                   results_path=results_path,
                                                   results_file='results' + file_name[0], iter_nr=r, bic_call=bic_call)
            log_likelihood_for_bic += cur_log_likelihood_for_bic
        log_likelihood_for_bic /= ITERATIONS

    plt.close('all')
    return log_likelihood_for_bic, N


def bic(filenames):
    # filenames = ['dataset1.txt', 'dataset2.txt', 'dataset3.txt']
    filename = filenames[0][0]
    print('Enter Kmax for ' + filename + ': ', end='')
    Kmax = int(input())
    d = 2  # The dimensions of the observations
    pms_per_k = 1 + d + d * (d + 1) / 2  # For alpha, means and covars

    bic_table = pd.DataFrame(data=None, index=np.arange(1, Kmax + 1), columns=['log_likelihood', 'BIC'])
    bic_table.index.name = 'K'
    for K in range(1, Kmax + 1):
        pk = K * pms_per_k
        log_likelihood, N = em(filenames, K, bic_call=True)
        bic_value = log_likelihood - (pk / 2) * np.log(N)
        bic_table.ix[K]['log_likelihood'] = log_likelihood
        bic_table.ix[K]['BIC'] = bic_value
    plt.close('all')
    print(filename)
    print('BIC table: ')
    print(bic_table)
