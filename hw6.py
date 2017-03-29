from EM_algorithm import em, bic
from K_means import kmeans

file_names = [[('dataset1.txt', 2)], [('dataset2.txt', 3)], [('dataset3.txt', 2)]]
# file_names = [('dataset1.txt', 2)]

for filename in file_names:
    kmeans(filename)
    em(filename)
    bic(filename)
