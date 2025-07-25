'''
This is the code of the paper
Jian Hou, Juntao Ge, Huaqiang Yuan, Marcello Pelillo. Experimental Evaluation of Szemer{\'e}di's Regularity Lemma in Graph-based Clustering. Pattern Recognition, 2025, doi:
https://doi.org/10.1016/j.patcog.2025.112205

The regularity partitioning part of this code is based on the codes of Marco Fiorucci in
https://github.com/MarcoFiorucci/dense_graph_reducer
https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions
'''

import numpy as np
from math import ceil
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import spectral_clustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import rand_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score

################################################# graph based clustering
def graph_cluster(method_c, sima, ncluster):
    if method_c == 'spc':
        label_c = spc2label(sima, ncluster)
    elif method_c == 'dset':
        label_c = dset2label(sima)
    elif method_c == 'apc':
        label_c = apc2label(sima)

    return label_c

######################################################## APC ##########################################################
def apc2label(sima):
    af = AffinityPropagation(affinity='precomputed', random_state = None).fit(sima)
    label_c = af.labels_

    return label_c

######################################################## SPC ##########################################################
def spc2label(sima, ncluster):
    ndata = sima.shape[0]
    if ndata > ncluster:
        label_c = spectral_clustering(sima, n_clusters=ncluster)
    else:
        label_c = np.zeros(ndata)

    return label_c

######################################################## DSet #########################################################
def dset2label(sima):
    label_c = dominant_sets(sima)

    return label_c

def dominant_sets(graph_mat, max_k=0, tol=1e-4, max_iter=500): 
    graph_cardinality = graph_mat.shape[0]
    if max_k == 0:
        max_k = graph_cardinality
    clusters = np.zeros(graph_cardinality)
    already_clustered = np.full(graph_cardinality, False, dtype=bool)

    for k in range(max_k):
        if graph_cardinality - already_clustered.sum() <= ceil(0.05 * graph_cardinality):
            break
        x = np.full(graph_cardinality, 1.0)
        x[already_clustered] = 0.0
        x /= x.sum()

        y = replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
        cluster = np.where(y >= 1.0 / (graph_cardinality * 1.5))[0]
        already_clustered[cluster] = True
        clusters[cluster] = k
    clusters[~already_clustered] = k
    return clusters

def replicator(A, x, inds, tol, max_iter):
    error = tol + 1.0
    count = 0
    while error > tol and count < max_iter:
        x_old = np.copy(x)
        for i in inds:
            x[i] = x_old[i] * (A[i] @ x_old)
        x /= np.sum(x)
        error = np.linalg.norm(x - x_old)
        count += 1
    return x

########################################## evaluation
def label2cqs(label_t, label_c):
    if np.min(label_t) < 1:
        label_t = label_t + 1 - np.min(label_t)

    if np.min(label_c) < 1:
        label_c = label_c + 1 - np.min(label_c)

    
    acc = cluster_accuracy(label_t, label_c)
    nmi = normalized_mutual_info_score(label_t, label_c)
    ari = adjusted_rand_score(label_t, label_c)
    ami = adjusted_mutual_info_score(label_t, label_c)
    rand = rand_score(label_t, label_c)
    vmeasure = v_measure_score(label_t, label_c)
    fmi = fowlkes_mallows_score(label_t, label_c)

    cq = clustering_quality(acc, ami, ari, nmi, rand, vmeasure, fmi)

    return cq

def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    D = D.astype(int)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class clustering_quality:
    acc = 0
    ami = 0
    ari = 0
    nmi = 0
    rand = 0
    vmeasure = 0
    fmi = 0

    def __init__(self, res_acc, res_ami, res_ari, res_nmi, res_rand, res_vmeansure, res_fmi):
        self.acc = res_acc
        self.ami = res_ami
        self.ari = res_ari
        self.nmi = res_nmi
        self.rand = res_rand
        self.vmeasure = res_vmeansure
        self.fmi = res_fmi
