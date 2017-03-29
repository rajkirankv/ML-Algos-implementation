import numpy as np
import scipy.stats as stats

SINGULARITY_EPSILON = 1e-4


def initialize_parameters(data, dims, K, type='random'):
    N = data.shape[0]

    seed_mean = np.mean(data, axis=0)
    seed_stdev = np.std(data, axis=0)
    means = seed_mean + np.random.randn(K, dims) * seed_stdev

    # Initializing covariance matrices is non-trivial. These matrices must be positive semi definite.
    # The method below tries to achieve that. However, more systematic methods can be explored.
    A = np.random.rand(dims, dims)
    variances = np.dot(A, A.T)
    # variances = np.random.chisquare(df=N, size=(K, dims, dims))
    # variances = np.random.rand(K, dims, dims)
    for k in range(1, K):
        A = np.random.rand(dims, dims)
        variance = np.dot(A, A.T)
        avoid_singularity(variance)
        variances = np.dstack((variances, variance))
    # variances = variances.reshape(K, dims, dims)
    variances = variances.T

    alphas = np.ones(K)
    weights = np.random.dirichlet(alpha=alphas, size=1).T
    return means, variances, weights


def avoid_singularity(matrix, threshold=SINGULARITY_EPSILON):
    sing_diag = np.diagonal(matrix) < threshold  # This is N * 1
    sing_diag = np.diag(sing_diag)  # This is a boolean N * N matrix that has True only on diagonal and only when
    # the diagonal element of the matrix is less than threshold
    if sing_diag.any():
        matrix[sing_diag] = threshold


def get_comp_distributions(D, means, covars):
    # D: N * d matrix
    # means: K * d matrix
    # covars: K * d * d matrix
    N = D.shape[0]
    mv_gaussian_pdf_matrix = np.empty(shape=(N, 0))
    for kth_mean, kth_covar in zip(means, covars):  # Loop through the K parameter matrices
        try:
            avoid_singularity(kth_covar)
            prob_from_kth_component = stats.multivariate_normal.pdf(D, kth_mean, kth_covar)  # This is a N * 1 vector
            # This may not always give reasonable probabilities owing to bad initializations of covar as mentioned in
            #  the corresponding function
        except (ValueError, np.linalg.linalg.LinAlgError) as e:
            prob_from_kth_component = np.zeros(shape=(N,))
        prob_from_kth_component = np.expand_dims(prob_from_kth_component, axis=1)
        mv_gaussian_pdf_matrix = np.hstack((mv_gaussian_pdf_matrix, prob_from_kth_component))

    return mv_gaussian_pdf_matrix  # This needs to be a N(observations) * K(components) matrix


def get_mixture_model(D, alphas, means, covars, probabilities_matrix=None):
    if probabilities_matrix is None:
        probabilities_matrix = get_comp_distributions(D, means=means, covars=covars)
    return np.dot(probabilities_matrix, alphas)  # This should be a N * 1 vector of probabilities
