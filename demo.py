'''
This is the code of the paper
Jian Hou, Juntao Ge, Huaqiang Yuan, Marcello Pelillo. Experimental Evaluation of Szemer{\'e}di's Regularity Lemma in Graph-based Clustering. Pattern Recognition, 2025, doi:
https://doi.org/10.1016/j.patcog.2025.112205

The regularity partitioning part of this code is based on the codes of Marco Fiorucci in
https://github.com/MarcoFiorucci/dense_graph_reducer
https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions
'''

import time
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from regularity_lemma import assign_outlier
from regularity_lemma import reduce_graph
from regularity_lemma import index_partition

from func import graph_cluster
from func import label2cqs

vec_sigma = [1] #[0.1, 0.2, 0.5, 1, 2, 5, 10]
vec_epsilon = [0.1, 0.2] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
vec_rate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
vec_b = [2, 3, 4, 5, 6, 7, 8, 9, 10]

method_c = 'spc'            # 'spc', 'apc', 'dset'

file_input = 'thyroid.txt'
data = np.loadtxt(file_input, delimiter=',')
X = data[:, :-1]
label_t = data[:, -1]
ncluster = len(np.unique(label_t))
ndata1 = len(label_t)

dima = euclidean_distances(X, X)

res_max = 0
for sigma in vec_sigma:
    sima = np.exp(- dima / (dima.mean() * sigma))

    ############################ straightforward clustering
    start_time_one_stage_clustering = time.time()
    label_c1 = graph_cluster(method_c, sima, ncluster)
    t1 = time.time() - start_time_one_stage_clustering

    cq1 = label2cqs(label_t, label_c1)
    ncluster1 = len(np.unique(label_c1))

    ############################### two-stage clustering
    count_parti = 0

    for epsilon in vec_epsilon:
        for rate in vec_rate:
            for b in vec_b:
                if b < ndata1:
                    start_time_two_stage_clustering = time.time()

                    #graph compression
                    reduced_sima, classes, nclass = reduce_graph(sima, epsilon, rate, b)

                    #clustering with reduced graph
                    reduced_labels = graph_cluster(method_c, reduced_sima, ncluster) + 1   #avoid 0 label, to avoid confusion with exceptional class

                    #mapping back to original graph
                    label_c2 = np.zeros(classes.shape)
                    for i in range(1, nclass + 1):
                        label_c2[classes == i] = reduced_labels[i - 1]

                    #assigning remaining points
                    label_c2 = assign_outlier(sima, label_c2)

                    t2 = time.time() - start_time_two_stage_clustering
                    cq2 = label2cqs(label_t, label_c2)

                    #other info
                    ind_parti = index_partition(sima, classes)
                    ncluster2 = len(np.unique(label_c2))
                    ndata2 = reduced_sima.shape[0]
                    node_size = len(np.where(classes == 1)[0])
                    idx_outlier = np.where(classes == 0)[0]       # data with classes ==0, outside the reduced graph
                    noutlier = len(idx_outlier)

                    if res_max < cq2.nmi:
                        res_max = cq2.nmi

                    print()
                    print('sigma = ' + str(sigma) + ' eps = ' + str(epsilon) + ' compression rate = ' + str(rate)  + ' b = ' + str(b))
                    print('ndata = ' + str(ndata1) + ' : ' + str(ndata2) + ' ncluster = ' + str(ncluster) + ' : ' + str(ncluster1) + ' : ' + str(ncluster2))
                    print("node size in reduced graph = " + str(node_size) + "  number of outliers = " + str(noutlier))

                    print('clustering result nmi one stage vs. two stage = ' + str(cq1.nmi) + ' : ' + str(cq2.nmi))
                    print('Computational Time = ' + str(t1) + ' : ' + str(t2) + ' ind_parti = ' + str(ind_parti))

                    print()
                    print('max = ' + str(res_max))

