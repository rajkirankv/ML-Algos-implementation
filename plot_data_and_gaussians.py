from plot_gauss_parameters import *


def plot_data_and_gaussians(data_file_name, means, covars):
    # load data
    # data = np.genfromtxt(data_file_name, delimiter=' ')
    data = np.genfromtxt(data_file_name)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # markers = ['o', 'p', '+', '.', '*']

    # plot data as a scatter plot
    P.scatter(data[:, 0], data[:, 1], s=5, c='k', marker='x', alpha=.65, linewidths=1)

    for mean, covar, color in zip(means, covars, colors):
        plot_gauss_parameters(mean, covar, color)

        # P.show()
        # return current_plot

# data_file_name = "/Users/stan/PycharmProjects/CS274a/HW6/data/dataset1.txt"
# data = np.genfromtxt(data_file_name, delimiter=' ')
# plot_data_and_gaussians(data_file_name, np.zeros((2,)), np.mean(data, 0), np.eye(2), np.eye(2))
