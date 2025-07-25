'''
This is the code of the paper
Jian Hou, Juntao Ge, Huaqiang Yuan, Marcello Pelillo. Experimental Evaluation of Szemer{\'e}di's Regularity Lemma in Graph-based Clustering. Pattern Recognition, 2025, doi:
https://doi.org/10.1016/j.patcog.2025.112205

The regularity partitioning part of this code is based on the codes of Marco Fiorucci in
https://github.com/MarcoFiorucci/dense_graph_reducer
https://github.com/MarcoFiorucci/graph-summarization-using-regular-partitions
'''

import numpy as np
import random

######################################################## main functions ###############################################

def reduce_graph(sima, epsilon, rate, b):
    m_parti = 'alon'
    m_init = 'degree'                            # degree, random
    m_refine = 'degree'                          # degree, indeg
    drop_edges_between_irregular_pairs = False

    alg = generate_szemeredi_reg_lemma_implementation(m_parti, sima, epsilon, m_init, m_refine, drop_edges_between_irregular_pairs)
    alg.run(b, rate)
    
    sima_reduced = alg.reduced_sim_mat
    classes = alg.classes
    nclass = alg.k

    return sima_reduced, classes, nclass

def assign_outlier(sima, label_c):
    idx_outlier = np.where(label_c == 0)[0]
    idx_inlier = np.where(label_c > 0)[0]
    sima1 = sima[idx_outlier, :]
    sima2 = sima1[:, idx_inlier]

    for i in np.arange(1, len(idx_outlier)+1, 1):
        vec = sima2[i - 1, :]
        idx_max =np.argmax(vec)
        label_c[idx_outlier[i - 1]] = label_c[idx_inlier[idx_max]]

    return label_c


########################################### supporting functions #############################################
def index_partition(sima, classes):
    label = np.unique(classes)
    k = len(label) - 1

    sum_den = 0
    for i in np.arange(1, k, 1):
        idx_a = np.where(classes == label[i])[0]
        for j in np.arange(i+1, k+1, 1):
            idx_b = np.where(classes == label[j])[0]
            den = edge_density(sima, idx_a, idx_b)
            sum_den = sum_den + den

    return sum_den / (k ** 2)


def edge_density(sima, indices_a, indices_b):
    if indices_a.size == 0 or indices_b.size == 0:
        return 0
    elif indices_a.size == indices_b.size == 1:
        return 0

    n_a = indices_a.size
    n_b = indices_b.size
    max_edges = n_a * n_b
    n_edges = sima[np.ix_(indices_a, indices_b)].sum()
    return n_edges / max_edges

def generate_szemeredi_reg_lemma_implementation(m_parti, sim_mat, epsilon, m_init, m_refine,
                                                drop_edges_between_irregular_pairs):
    """
    generate an implementation of the Szemeredi regularity lemma for the graph summarization
    :param kind: the kind of implementation to generate. The currently accepted strings are 'alon' for the Alon
                 the Alon implementation, and 'frieze_kannan' for the Frieze and Kannan implementation
    :param sim_mat: the similarity matrix representing the graph
    :param epsilon: the epsilon parameter to determine the regularity of the partition
    :param is_weighted: set it to True to specify if the graph is weighted
    :param random_initialization: set it to True to perform to generate a random partition of the graph
    :param random_refinement: set it to True to randomly re-asset the nodes in the refinement step
    :param is_fully_connected_reduced_matrix: if set to True the similarity matrix is not thresholded and a fully
           connected graph is generated
    :param is_no_condition_considered_regular: if set to True, when no condition
    :return:
    """

    is_weighted = True        #consider only weighted graph for clustering here
    # alg = srl.SzemerediRegularityLemma(sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs)
    alg = SzemerediRegularityLemma(sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs)

    if m_init == 'degree':
        alg.partition_initialization = initialization_degree_based  # consider add () after the function name ?
    elif m_init == 'random':
        alg.partition_initialization = initialization_random  # move partition_initialization in this file, and change to partition_initialization_random ...


    if m_refine == 'degree':
        alg.refinement_step = refinement_degree_based
    elif m_refine == 'index':
        alg.refinement_step = refinement_indeg_guided  # refinement_step.random

    if m_parti == "alon":
        alg.conditions = [condi_alon1, condi_alon2, condi_alon3]
    elif m_parti == "frieze_kannan":
        alg.conditions = [condi_frieze_kannan]
    else:
        raise ValueError("Could not find the specified graph summarization method")

    return alg

######################################## partition initialization #################################################

def initialization_random(self, b=2):
    """
    perform step 1 of Alon algorithm,  initializing the starting partition of a graph G.
    The result is stored in "classes" attribute
    :param  G: the similarity matrix or adjacency matrix of G
    :param  b: the number of classes in which the vertices are partitioned
    :param  random: if set to True, the vertices are assigned randomly to the classes,
            if set to False, node are decreasingly ordered by their degree and then splitted
            in classes
    """
    self.k = b
    self.classes = np.zeros(self.N)
    self.classes_cardinality = self.N // self.k

    for i in range(self.k):
        self.classes[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)] = i + 1

    np.random.shuffle(self.classes)

    print('random initialization')


def initialization_degree_based(self, b=2):
    """
    perform step 1 of Alon algorithm,  initializing the starting partition of a graph G.
    The result is stored in "classes" attribute
    :param  G: the similarity matrix or adjacency matrix of G
    :param  b: the number of classes in which the vertices are partitioned
    :param  random: if set to True, the vertices are assigned randomly to the classes,
            if set to False, node are decreasingly ordered by their degree and then splitted
            in classes
    """
    self.k = b
    self.classes = np.zeros(self.N)
    self.classes_cardinality = self.N // self.k

    for i in range(self.k):
        self.classes[self.degrees[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)]] = i + 1

################################################## refinement step #####################################################

def refinement_random_based(self):
    """ Perform step 4 of Alon algorithm, performing the refinement of the pairs, processing nodes in a random way. Some heuristic is applied in order to speed up the process.
    """
    pass


def partition_correct(self):
    """ Checks if the partition cardinalities are valid
    :returns: True if the classes of the partition have the right cardinalities, false otherwise
    """
    for i in range(1, self.k+1):
        if not np.where(self.classes == i)[0].size == self.classes_cardinality:
            return False
    return True



##########################################################################
######################## INDEGREE REFINEMENT #############################
##########################################################################

#edge density
def density(self, indices_a, indices_b):
    """ Calculates the density between two sets of vertices
    :param indices_a: np.array(), the indices of the first set
    :param indices_b: np.array(), the indices of the second set
    """
    if indices_a.size == 0 or indices_b.size == 0:
        return 0
    elif indices_a.size == indices_b.size == 1:
        return 0

    # [TODO] performance issue: comparing all the indices? maybe add a parameter to the function
    if np.array_equal(indices_a, indices_b):
        n = indices_a.size
        max_edges = (n*(n-1))/2
        n_edges = np.tril(self.adj_mat[np.ix_(indices_a, indices_a)], -1).sum()
        #n_edges = np.tril(self.sim_mat[np.ix_(indices_a, indices_a)], -1).sum()
        return n_edges / max_edges

    n_a = indices_a.size
    n_b = indices_b.size
    max_edges = n_a * n_b
    n_edges = self.adj_mat[np.ix_(indices_a, indices_b)].sum()
    #n_edges = self.sim_mat[np.ix_(indices_a, indices_b)].sum()
    return n_edges / max_edges


def compute_indensities(self):
    """ Compute the inside density for each class of a given partition
    :returns: np.array(float32) of densities for each class in the partition
    """
    cls = list(range(0, self.k + 1))
    densities = np.zeros(len(cls), dtype='float32')
    for c in cls:
        c_indices = np.where(self.classes == c)[0]
        if c_indices.size:
            densities[c] = density(self, c_indices, c_indices)
        else:
            densities[c] = 0

    return densities


def choose_candidate(self, in_densities, s, irregulars):
    """ This function chooses a class between the irregular ones (d(ci,cj), 1-|d(ci,ci)-d(cj,cj)|)
    :param in_densities: list(float), precomputed densities to speed up the calculations
    :param s: int, the class which all the other classes are compared to
    :param irregulars: list(int), the list of irregular classes
    """
    candidate_idx = -1
    candidate = -1

    # Exploit the precalculated densities
    s_dens = in_densities[s]
    for r in irregulars:
        s_indices = np.where(self.classes == s)[0]
        r_indices = np.where(self.classes == r)[0]
        r_idx = density(self, s_indices, r_indices) + (1 - abs(s_dens - in_densities[r]))
        if r_idx > candidate_idx:
            candidate_idx = r_idx
            candidate = r

    return candidate


def fill_new_set(self, new_set, compls, maximize_density):
    """ Find nodes that can be added
    Move from compls the nodes in can_be_added until we either finish the nodes or reach the desired cardinality
    :param new_set: np.array(), array of indices of the set that must be augmented
    :param compls: np.array(), array of indices used to augment the new_set
    :param maximize_density: bool, used to augment or decrement density
    """

    if maximize_density:
        nodes = self.adj_mat[np.ix_(new_set, compls)] == 1.0
        #nodes = self.sim_mat[np.ix_(new_set, compls)] >= 0.5

        # These are the nodes that can be added to certs, we take the most connected ones with all the others
        to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes], return_counts=True)
        to_add = to_add[0][to_add[1].argsort()]
    else:
        nodes = self.adj_mat[np.ix_(new_set, compls)] == 0.0
        #nodes = self.sim_mat[np.ix_(new_set, compls)] < 0.5

        # These are the nodes that can be added to certs, we take the less connected ones with all the others
        to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes], return_counts=True)
        to_add = to_add[0][to_add[1].argsort()[::-1]]

    while new_set.size < self.classes_cardinality:

        # If there are nodes in to_add, we keep moving from compls to new_set
        if to_add.size > 0:
            node, to_add = to_add[-1], to_add[:-1]
            new_set = np.append(new_set, node)
            compls = np.delete(compls, np.argwhere(compls == node))

        else:
            # If there aren't candidate nodes, we keep moving from complements
            # to certs until we reach the desired cardinality
            node, compls = compls[-1], compls[:-1]
            new_set = np.append(new_set, node)

    return new_set, compls


def refinement_indeg_guided(self):
    """ In-degree based refinemet. The refinement exploits the internal structure of the classes of a given partition.
    :returns: True if the new partition is valid, False otherwise
    """
    #ipdb.set_trace()
    threshold = 0.5

    to_be_refined = list(range(1, self.k + 1))
    old_cardinality = self.classes_cardinality
    self.classes_cardinality //= 2
    in_densities = compute_indensities(self)
    new_k = 0

    # print('step indeg')

    while to_be_refined:
        s = to_be_refined.pop(0)
        irregular_r_indices = []

        for r in to_be_refined:
            if self.certs_compls_list[r - 2][s - 1][0][0]:
                irregular_r_indices.append(r)

        # If class s has irregular classes
        if irregular_r_indices:

            # Choose candidate based on the inside-outside density index
            r = choose_candidate(self, in_densities, s, irregular_r_indices)
            to_be_refined.remove(r)

            s_certs = np.array(self.certs_compls_list[r - 2][s - 1][0][1]).astype('int32')
            s_compls = np.array(self.certs_compls_list[r - 2][s - 1][1][1]).astype('int32')
            assert s_certs.size + s_compls.size == old_cardinality

            r_compls = np.array(self.certs_compls_list[r - 2][s - 1][1][0]).astype('int32')
            r_certs = np.array(self.certs_compls_list[r - 2][s - 1][0][0]).astype('int32')
            assert r_certs.size + r_compls.size == old_cardinality


            # Merging the two complements
            compls = np.append(s_compls, r_compls)

            # Calculating certificates densities
            dens_s_cert = density(self, s_certs, s_certs)
            dens_r_cert = density(self, r_certs, r_certs)

            for cert, dens in [(s_certs, dens_s_cert), (r_certs, dens_r_cert)]:

                # Indices of the cert ordered by in-degree, it doesn't matter if we reverse the list as long as we unzip it
                degs = self.adj_mat[np.ix_(cert, cert)].sum(1).argsort()[::-1]
                #degs = self.sim_mat[np.ix_(cert, cert)].sum(1).argsort()[::-1]

                if dens > threshold:
                    # Certificates high density branch

                    # Unzip them in half to preserve seeds
                    set1=  cert[degs[0:][::2]]
                    set2 =  cert[degs[1:][::2]]

                    # Adjust cardinality of the new set to the desired cardinality
                    set1, compls = fill_new_set(self, set1, compls, True)
                    set2, compls = fill_new_set(self, set2, compls, True)

                    # Handling of odd classes
                    new_k -= 1
                    self.classes[set1] = new_k
                    if set1.size > self.classes_cardinality:
                        self.classes[set1[-1]] = 0
                    new_k -= 1
                    self.classes[set2] = new_k
                    if set2.size > self.classes_cardinality:
                        self.classes[set2[-1]] = 0

                else:
                    # Certificates low density branch
                    set1 = np.random.choice(cert, len(cert)//2, replace=False)
                    set2 = np.setdiff1d(cert, set1)

                    # Adjust cardinality of the new set to the desired cardinality
                    set1, compls = fill_new_set(self, set1, compls, False)
                    set2, compls = fill_new_set(self, set2, compls, False)

                    # Handling of odd classes
                    new_k -= 1
                    self.classes[set1] = new_k
                    if set1.size > self.classes_cardinality:
                        self.classes[set1[-1]] = 0
                    new_k -= 1
                    self.classes[set2] = new_k
                    if set2.size > self.classes_cardinality:
                        self.classes[set2[-1]] = 0

                # Handle special case when there are still some complements not assigned
                if compls.size > 0:
                    self.classes[compls] = 0

        else:
            # The class is e-reg with all the others or it does not have irregular classes

            # Sort by indegree and unzip the structure
            s_indices = np.where(self.classes == s)[0]
            s_indegs = self.adj_mat[np.ix_(s_indices, s_indices)].sum(1).argsort()
            #s_indegs = self.sim_mat[np.ix_(s_indices, s_indices)].sum(1).argsort()

            set1=  s_indices[s_indegs[0:][::2]]
            set2=  s_indices[s_indegs[1:][::2]]

            # Handling of odd classes
            new_k -= 1
            self.classes[set1] = new_k
            if set1.size > self.classes_cardinality:
                self.classes[set1[-1]] = 0
            new_k -= 1
            self.classes[set2] = new_k
            if set1.size > self.classes_cardinality:
                self.classes[set1[-1]] = 0

    # print('key step')
    self.k *= 2
    # print('k = ' + str(self.k))

    # Check validity of class C0, if invalid and enough nodes, distribute the exceeding nodes among the classes
    c0_indices = np.where(self.classes == 0)[0]
    if c0_indices.size >= (self.epsilon * self.adj_mat.shape[0]):
        if c0_indices.size > self.k:
            self.classes[c0_indices[:self.k]] = np.array(range(1, self.k+1))*-1
        else:
            print('[ refinement ] Invalid cardinality of C_0')
            return False

    self.classes *= -1

    if not partition_correct(self):
        ipdb.set_trace()
    return True


##########################################################################
######################## PAIR DEGREE REFINEMENT ##########################
##########################################################################


def within_degrees(self, c):
    """ Given a class c it returns the degrees calculated within the class
    :param c: int, class c
    :returns: np.array(int16), list of n indices where the indices in c have the in-degree
    """
    c_degs = np.zeros(len(self.degrees), dtype='int16')
    c_indices = np.where(self.classes == c)[0]
    c_degs[c_indices] = self.adj_mat[np.ix_(c_indices, c_indices)].sum(1)

    return c_degs


def get_s_r_degrees(self,s,r):
    """ Given two classes it returns a degree vector (indicator vector) where the degrees
    have been calculated with respecto to each other set.
    :param s: int, class s
    :param r: int, class r
    :returns: np.array, degree vector
    """

    s_r_degs = np.zeros(len(self.degrees), dtype='int16')

    # Gets the indices of elements which are part of class s, then r
    s_indices = np.where(self.classes == s)[0]
    r_indices = np.where(self.classes == r)[0]

    # Calculates the degree and assigns it
    s_r_degs[s_indices] = self.adj_mat[np.ix_(s_indices, r_indices)].sum(1)
    s_r_degs[r_indices] = self.adj_mat[np.ix_(r_indices, s_indices)].sum(1)

    return s_r_degs



def refinement_degree_based(self):
    """
    perform step 4 of Alon algorithm, performing the refinement of the pairs, processing nodes according to their degree. Some heuristic is applied in order to
    speed up the process
    """
    #ipdb.set_trace()
    to_be_refined = list(range(1, self.k + 1))
    irregular_r_indices = []
    is_classes_cardinality_odd = self.classes_cardinality % 2 == 1
    self.classes_cardinality //= 2

    while to_be_refined:
        s = to_be_refined.pop(0)

        for r in to_be_refined:
            if self.certs_compls_list[r - 2][s - 1][0][0]:
                irregular_r_indices.append(r)

        if irregular_r_indices:
            np.random.seed(314)
            random.seed(314)
            chosen = random.choice(irregular_r_indices)
            to_be_refined.remove(chosen)
            irregular_r_indices = []

            # Degrees wrt to each other class
            s_r_degs = get_s_r_degrees(self, s, chosen)

            # i = 0 for r, i = 1 for s
            for i in [0, 1]:
                cert_length = len(self.certs_compls_list[chosen - 2][s - 1][0][i])
                compl_length = len(self.certs_compls_list[chosen - 2][s - 1][1][i])

                greater_set_ind = np.argmax([cert_length, compl_length])
                lesser_set_ind = np.argmin([cert_length, compl_length]) if cert_length != compl_length else 1 - greater_set_ind

                greater_set = self.certs_compls_list[chosen - 2][s - 1][greater_set_ind][i]
                lesser_set = self.certs_compls_list[chosen - 2][s - 1][lesser_set_ind][i]

                self.classes[lesser_set] = 0

                difference = len(greater_set) - self.classes_cardinality
                # retrieve the first <difference> nodes sorted by degree.
                # N.B. NODES ARE SORTED IN DESCENDING ORDER
                difference_nodes_ordered_by_degree = sorted(greater_set, key=lambda el: s_r_degs[el], reverse=True)[0:difference]
                #difference_nodes_ordered_by_degree = sorted(greater_set, key=lambda el: np.where(self.degrees == el)[0], reverse=True)[0:difference]

                self.classes[difference_nodes_ordered_by_degree] = 0
        else:
            self.k += 1
            # print('k = ' + str(self.k))
            #  TODO: cannot compute the r_s_degs since the candidate does not have any e-regular pair  <14-11-17, lakj>
            s_indices_ordered_by_degree = sorted(list(np.where(self.classes == s)[0]), key=lambda el: np.where(self.degrees == el)[0], reverse=True)
            #s_indices_ordered_by_degree = sorted(list(np.where(self.classes == s)[0]), key=lambda el: s_r_degs[el], reverse=True)

            if is_classes_cardinality_odd:
                self.classes[s_indices_ordered_by_degree.pop(0)] = 0
            self.classes[s_indices_ordered_by_degree[0:self.classes_cardinality]] = self.k

    C0_cardinality = np.sum(self.classes == 0)
    num_of_new_classes = C0_cardinality // self.classes_cardinality
    nodes_in_C0_ordered_by_degree = np.array([x for x in self.degrees if x in np.where(self.classes == 0)[0]])
    for i in range(num_of_new_classes):
        self.k += 1
        # print('kk = ' + str(self.k))
        self.classes[nodes_in_C0_ordered_by_degree[
                     (i * self.classes_cardinality):((i + 1) * self.classes_cardinality)]] = self.k

    C0_cardinality = np.sum(self.classes == 0)
    if C0_cardinality > self.epsilon * self.N:
        #sys.exit("Error: not enough nodes in C0 to create a new class.Try to increase epsilon or decrease the number of nodes in the graph")
        #print("Error: not enough nodes in C0 to create a new class. Try to increase epsilon or decrease the number of nodes in the graph")

        if not partition_correct(self):
            ipdb.set_trace()
        return False

    if not partition_correct(self):
        ipdb.set_trace()
    return True

############################################### Conditions ###########################################################

def condi_alon1(self, cl_pair):
    """
    verify the first condition of Alon algorithm (regularity of pair)
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    :return: A list of two empty lists representing the empty certificates
    :return: A list of two empty lists representing the empty complements
    """
    return cl_pair.bip_avg_deg < (self.epsilon ** 3.0) * cl_pair.n, [[], []], [[], []]


def condi_alon2(self, cl_pair):
    """
    verify the second condition of Alon algorithm (irregularity of pair)
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """
    certs = []
    compls = []
    s_vertices_degrees = cl_pair.classes_vertices_degrees()[1, :]
    deviated_nodes = np.abs(s_vertices_degrees - cl_pair.bip_avg_deg) > (self.epsilon ** 4.0) * cl_pair.n
    deviation_threshold = (self.epsilon ** 4.0) * cl_pair.n
    # retrieve positive deviated nodes
    one_direction_nodes = deviated_nodes * (s_vertices_degrees - cl_pair.bip_avg_deg > deviation_threshold)
    is_irregular = one_direction_nodes.sum() >= (1.0 / 16.0) * (self.epsilon ** 4.0) * cl_pair.n

    if is_irregular:
        certs.append(list(cl_pair.index_map[0][range(cl_pair.n)]))
        certs.append(list(cl_pair.index_map[1][one_direction_nodes]))
        compls.append([])
        compls.append(list(cl_pair.index_map[1][~one_direction_nodes]))
    else:
        # retrieve negative deviated nodes
        one_direction_nodes = deviated_nodes * (s_vertices_degrees - cl_pair.bip_avg_deg < -deviation_threshold)
        is_irregular = one_direction_nodes.sum() >= (1.0 / 16.0) * (self.epsilon ** 4.0) * cl_pair.n
        if is_irregular:
            certs.append(list(cl_pair.index_map[0][range(cl_pair.n)]))
            certs.append(list(cl_pair.index_map[1][one_direction_nodes]))
            compls.append([])
            compls.append(list(cl_pair.index_map[1][~one_direction_nodes]))
        else:
            certs = [[], []]
            compls = [[], []]

    return is_irregular, certs, compls


def condi_alon3(self, cl_pair, fast_convergence=True):
    """
    verify the third condition of Alon algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :param fast_convergence: apply the fast convergence version of condition 3
    :return: True if the condition is verified, False otherwise
    """

    is_irregular = False

    cert_s = []
    compl_s = []
    y0 = -1

    # nh_mat = cl_pair.neighbourhood_matrix()
    # nh_dev_mat = cl_pair.neighbourhood_deviation_matrix(nh_mat)
    # s_degrees = np.diag(nh_mat)
    nh_dev_mat, s_degrees = cl_pair.neighbourhood_deviation_matrix()
    if fast_convergence:
        Y_indices = cl_pair.find_Y(nh_dev_mat)

        if not list(Y_indices):
            # enter in Y spurious condition
            is_irregular = True
            return is_irregular, [[], []], [[], []]

        Y_degrees = s_degrees[Y_indices]
        Yp_indices = cl_pair.find_Yp(Y_degrees, Y_indices)

        if not list(Yp_indices):
            # enter in Yp spurious condition
            is_irregular = False
            return is_irregular, [[], []], [[], []]

        y0 = cl_pair.compute_y0(nh_dev_mat, Y_indices, Yp_indices)

        cert_s, compl_s = cl_pair.find_s_cert_and_compl(nh_dev_mat, y0, Yp_indices)
    else:
        s_indices = cl_pair.find_Yp(s_degrees, np.arange(cl_pair.n))

        for y0 in s_indices:
            cert_s, compl_s = cl_pair.find_s_cert_and_compl(nh_dev_mat, y0, s_indices)
            if cert_s:
                break
    cert_r, compl_r = cl_pair.find_r_cert_and_compl(y0)

    if cert_r and cert_s:
        is_irregular = True
    else:
        cert_r = []
        cert_s = []
        compl_r = []
        compl_s = []

    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]


def condi_frieze_kannan(self, cl_pair):
    """
    verify the condition of Frieze and Kannan algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """
    cert_r = []
    cert_s = []
    compl_r = []
    compl_s = []

    if self.is_weighted:
        W = cl_pair.bip_sim_mat - cl_pair.bip_density
    else:
        W = cl_pair.bip_adj_mat - cl_pair.bip_density

    x, sv_1, y = scipy.sparse.linalg.svds(W, k=1, which='LM')

    is_irregular = (sv_1 >= self.epsilon * cl_pair.n)

    if is_irregular:
        beta = 3.0 / self.epsilon
        x = x.ravel()
        y = y.ravel()
        hat_thresh = beta / math.sqrt(cl_pair.n)
        x_hat = np.where(np.abs(x) <= hat_thresh, x, 0.0)
        y_hat = np.where(np.abs(y) <= hat_thresh, y, 0.0)

        quadratic_threshold = (self.epsilon - 2.0 / beta) * (cl_pair.n / 4.0)

        x_mask = x_hat > 0
        y_mask = y_hat > 0
        x_plus = np.where(x_mask, x_hat, 0.0)
        x_minus = np.where(~x_mask, x_hat, 0.0)
        y_plus = np.where(y_mask, y_hat, 0.0)
        y_minus = np.where(~y_mask, y_hat, 0.0)

        r_mask = np.empty((0, 0))
        s_mask = np.empty((0, 0))

        q_plus = y_plus * 1.0 / hat_thresh
        q_minus = y_minus * 1.0 / hat_thresh

        if x_plus @ W @ y_plus >= quadratic_threshold:
            r_mask = (W @ q_plus) >= 0.0
            s_mask = (r_mask @ W) >= 0.0
        elif x_plus @ W @ y_minus >= quadratic_threshold:
            r_mask = (W @ q_minus) >= 0.0
            s_mask = (r_mask @ W) <= 0.0
        elif x_minus @ W @ y_plus >= quadratic_threshold:
            r_mask = (W @ q_plus) <= 0.0
            s_mask = (r_mask @ W) >= 0.0
        elif x_minus @ W @ y_minus >= quadratic_threshold:
            r_mask = (W @ q_minus) <= 0.0
            s_mask = (r_mask @ W) <= 0.0
        else:
            sys.exit("no condition on the quadratic form was verified")

        cert_r = list(cl_pair.index_map[0][r_mask])
        compl_r = list(cl_pair.index_map[0][~r_mask])
        cert_s = list(cl_pair.index_map[1][s_mask])
        compl_s = list(cl_pair.index_map[1][~s_mask])
    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]

############################################## the key classes #######################################################

class SzemerediRegularityLemma:
    # methods required by the algorithm. User can provide its own implementations provided that they respect the input/output conventions
    partition_initialization = None
    """The method used to build the initial partition"""
    refinement_step = None
    """The method used to refine the current partition"""
    conditions = []
    """The conditions used to check the regularity/irregularity of the pairs"""

    # main data structure
    sim_mat = np.empty((0, 0))
    """The similarity matrix representing the graph (used only if is_weighted is set to True)"""
    adj_mat = np.empty((0, 0))
    """The adjacency matrix representing the graph"""
    reduced_sim_mat = np.empty((0, 0))
    """the resulting similarity matrix"""
    classes = np.empty((0, 0))
    """array with size equal to the number of nodes in the graph. Each element is set to the class whose node belongs"""
    degrees = np.empty((0, 0))
    """array containing the indices of the nodes ordered by the degree"""

    # main parameters of the algorithm
    N = 0
    """number of nodes in the graph"""
    k = 0
    """number of classes composing the partition"""
    classes_cardinality = 0
    """cardinality of each class"""
    epsilon = 0.0
    """epsilon parameter"""
    index_vec = []
    """index measuring the goodness of the partition"""

    # auxiliary structures used to keep track of the relation between the classes of the partition
    certs_compls_list = []
    """structure containing the certificates and complements for each pair in the partition"""
    regularity_list = []
    """list of lists of size k, each element i contains the list of classes regular with class i"""

    # flags to specify different behaviour of the algorithm
    is_weighted = True
    """flag to specify if the graph is weighted or not"""
    drop_edges_between_irregular_pairs = False
    """flag to specify if the reduced matrix is fully connected or not"""

    # debug structures
    condition_verified = []
    """this attribute is kept only for analysis purposes. For each iteration it stores the number of times that
       condition 1/condition 2/condition 3/no condition has been verified"""

    def __init__(self, sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs):
        if is_weighted:
            self.sim_mat = sim_mat
        self.adj_mat = (sim_mat > 0.0).astype(float)
        self.epsilon = epsilon
        self.N = self.adj_mat.shape[0]
        self.degrees = np.argsort(self.adj_mat.sum(0))

        # flags
        self.is_weighted = is_weighted
        self.drop_edges_between_irregular_pairs = drop_edges_between_irregular_pairs

    def generate_reduced_sim_mat(self):
        """
        generate the similarity matrix of the current classes
        :return sim_mat: the reduced similarity matrix
        """
        self.reduced_sim_mat = np.zeros((self.k, self.k))

        for r in range(2, self.k + 1):
            for s in (range(1, r) if not self.drop_edges_between_irregular_pairs else self.regularity_list[r - 2]):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)
                self.reduced_sim_mat[r - 1, s - 1] = self.reduced_sim_mat[s - 1, r - 1] = cl_pair.bip_density

    def reconstruct_original_mat(self, thresh, intracluster_weight=0):
        """
        reconstruct a similarity matrix with size equals to the original one, from the reduced similarity matrix
        :param thresh: a threshold parameter to prune the edges of the graph
        :param intracluster_weight: the weight to assign at each connection generated by the expansion of a cluster
        :return: the reconstructed graph
        """
        reconstructed_mat = np.zeros((self.N, self.N))

        r_nodes = self.classes > 0

        reconstructed_mat[np.ix_(r_nodes, r_nodes)] = intracluster_weight

        for r in range(2, self.k + 1):
            r_nodes = self.classes == r
            reconstructed_mat[np.ix_(r_nodes, r_nodes)] = intracluster_weight
            for s in range(1, r):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)

                s_nodes = self.classes == s
                if cl_pair.bip_density > thresh:
                    reconstructed_mat[np.ix_(r_nodes, s_nodes)] = reconstructed_mat[np.ix_(s_nodes, r_nodes)] = cl_pair.bip_density
        np.fill_diagonal(reconstructed_mat, 0.0)
        return reconstructed_mat

    def check_pairs_regularity(self):
        """
        perform step 2 of Alon algorithm, determining the regular/irregular pairs and their certificates and complements
        :return certs_compls_list: a list of lists containing the certificate and complement
                for each pair of classes r and s (s < r). If a pair is epsilon-regular
                the corresponding complement and certificate in the structure will be set to the empty lists
        :return num_of_irregular_pairs: the number of irregular pairs
        """
        # debug structure
        self.condition_verified = [0] * (len(self.conditions) + 1)

        num_of_irregular_pairs = 0
        index = 0.0

        for r in range(2, self.k + 1):
            self.certs_compls_list.append([])
            self.regularity_list.append([])

            for s in range(1, r):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)

                is_verified = False
                for i, cond in enumerate(self.conditions):
                    is_verified, cert_pair, compl_pair = cond(self, cl_pair)
                    if is_verified:
                        self.certs_compls_list[r - 2].append([cert_pair, compl_pair])

                        if cert_pair[0]:
                            num_of_irregular_pairs += 1
                        else:
                            self.regularity_list[r - 2].append(s)

                        self.condition_verified[i] += 1
                        break

                if not is_verified:
                    # if no condition was verified then consider the pair to be regular
                    self.certs_compls_list[r - 2].append([[[], []], [[], []]])
                    self.condition_verified[-1] += 1

                index += cl_pair.compute_bip_density() ** 2.0

        index *= (1.0 / self.k ** 2.0)
        self.index_vec.append(index)
        return num_of_irregular_pairs

    def check_partition_regularity(self, num_of_irregular_pairs):
        """
        perform step 3 of Alon algorithm, checking the regularity of the partition
        :param num_of_irregular_pairs: the number of found irregular pairs in the previous step
        :return: True if the partition is irregular, False otherwise
        """
        return num_of_irregular_pairs <= self.epsilon * ((self.k * (self.k - 1)) / 2.0)

    def run(self, b=2, compression_rate=0.05, iteration_by_iteration=False, verbose=False):
        """
        run the Alon algorithm.
        :param b: the cardinality of the initial partition (C0 excluded)
        :param compression_rate: the minimum compression rate granted by the algorithm, if set to a value in (0.0, 1.0]
                                 the algorithm will stop when k > int(compression_rate * |V|). If set to a value > 1.0
                                 the algorithm will stop when k > int(compression_rate)
        :param iteration_by_iteration: if set to true, the algorithm will wait for a user input to proceed to the next
               one
        :param verbose: if set to True some debug info is printed
        :return the reduced similarity matrix
        """
        # np.random.seed(314)
        # random.seed(314)

        if 0.0 < compression_rate <= 1.0:
            max_k = int(compression_rate * self.N)
        elif compression_rate > 1.0:
            max_k = int(compression_rate)
        # else:
        #     raise ValueError("incorrect compression rate. Only float greater than 0.0 are accepted")

        iteration = 0
        if verbose:
            print("Performing partition initialization")
        self.partition_initialization(self, b)
        while True:
            self.certs_compls_list = []
            self.regularity_list = []
            self.condition_verified = [0] * len(self.conditions)
            iteration += 1
            if verbose:
                print("Iteration " + str(iteration))
                print("Performing pairs regularity check")
            num_of_irregular_pairs = self.check_pairs_regularity()
            # print('step 1: ' + str(num_of_irregular_pairs))
            if verbose:
                total_pairs = (self.k * (self.k - 1)) / 2.0
                print("irregular pairs / total pairs = " + str(num_of_irregular_pairs) + " / " + str(int(total_pairs)))
                print("irregular pairs ratio = " + str(num_of_irregular_pairs / (self.epsilon * total_pairs)))
                print("k = " + str(self.k) + ". Class cardinality = " + str(
                    self.classes_cardinality) + ". Index = " + str(self.index_vec[-1]))
                print("conditions verified = " + str(self.condition_verified))

                print("Performing partition regularity check")
            if self.check_partition_regularity(num_of_irregular_pairs):
                if verbose:
                    print("The partition is regular")
                break

            if compression_rate > 0:                 # revised here
                if self.k >= max_k:
                    if verbose:
                        print("Either the classes cardinality is too low or the number of classes is too high. "
                              "Stopping iterations")
                    break
            if verbose:
                print("The partition is irregular, proceed to refinement")
                print("Performing refinement")
            self.refinement_step(self)

            #print(self.k)

            if iteration_by_iteration:
                input("Press Enter to continue...")
            if verbose:
                print()
        self.generate_reduced_sim_mat()
        return self.reduced_sim_mat

################################################ classes pair ########################################################
class ClassesPair:
    bip_adj_mat = np.empty((0, 0))
    """The bipartite adjacency matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s"""
    r = s = -1
    """The classes composing the bipartite graph"""
    n = 0
    """the cardinality of a class"""
    index_map = np.empty((0, 0))
    """A mapping from the bipartite adjacency matrix nodes to the adjacency matrix ones"""
    bip_avg_deg = 0
    """the average degree of the graph"""
    bip_density = 0
    """the average density of the graph"""
    epsilon = 0.0
    """the epsilon parameter"""

    def __init__(self, adj_mat, classes, r, s, epsilon):
        self.r = r
        self.s = s
        self.index_map = np.where(classes == r)[0]
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.n = self.bip_adj_mat.shape[0]
        self.bip_avg_deg = self.bip_avg_degree()
        self.bip_density = self.compute_bip_density()
        self.epsilon = epsilon

    def bip_avg_degree(self):
        """
        compute the average degree of the bipartite graph
        :return the average degree
        """
        return (self.bip_adj_mat.sum(0) + self.bip_adj_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        """
        compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the
        bipartite graph
        :return the density
        """
        return float(self.bip_adj_mat.sum()) / (self.n ** 2.0)

    def classes_vertices_degrees(self):
        """
        compute the degree of all vertices in the bipartite graph
        :return a (n,) numpy array containing the degree of each vertex
        """
        c_v_degs = np.sum(self.bip_adj_mat, 0)
        c_v_degs = np.vstack((c_v_degs, np.sum(self.bip_adj_mat, 1)))
        return c_v_degs

    # def neighbourhood_matrix(self, transpose_first=True):
    #     if transpose_first:
    #         return self.bip_adj_mat.T @ self.bip_adj_mat
    #     else:
    #         return self.bip_adj_mat @ self.bip_adj_mat.T
    #
    # def neighbourhood_deviation_matrix(self, nh_mat):
    #     return nh_mat - ((self.bip_avg_deg ** 2.0) / self.n)

    def neighbourhood_deviation_matrix(self, transpose_first=True):
        if transpose_first:
            mat = self.bip_adj_mat.T @ self.bip_adj_mat
        else:
            mat = self.bip_adj_mat @ self.bip_adj_mat.T
        rs_degrees = np.diag(mat)
        mat -= (self.bip_avg_deg ** 2.0) / self.n
        return mat, rs_degrees

    def find_Y(self, nh_dev_mat):
        inner_sums = nh_dev_mat.sum(1) - np.diag(nh_dev_mat)
        inner_sums_indices = np.argsort(inner_sums)[::-1]
        y_card_thresh = int((self.epsilon * self.n) + 1)
        outer_sum = inner_sums[inner_sums_indices[0:(y_card_thresh - 1)]].sum()

        for i in range(y_card_thresh, self.n):
            outer_sum += inner_sums[inner_sums_indices[i]]
            sigma_y = outer_sum / (i ** 2.0)
            # print "sigma_y = " + str(sigma_y)
            if sigma_y >= ((self.epsilon ** 3.0) / 2.0) * self.n:
                return inner_sums_indices[0:i]
        return np.array([])

    def find_Yp(self, degrees, Y_indices):
        # print "min el = " + str(np.min(degrees - self.bip_avg_degree))
        return Y_indices[np.abs(degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)]

    def compute_y0(self, nh_dev_mat, Y_indices, Yp_indices):
        sums = np.full((self.n,), -np.inf)
        for i in Yp_indices:
            sums[i] = 0
            for j in list(set(Y_indices) - set(Yp_indices)):
                sums[i] += nh_dev_mat[i, j]
        return np.argmax(sums)

    def find_s_cert_and_compl(self, nh_dev_mat, y0, Yp_indices):
        outliers_in_s = set(np.where(nh_dev_mat[y0, :] > 2.0 * (self.epsilon ** 4.0) * self.n)[0])
        outliers_in_Yp = list(set(Yp_indices) & outliers_in_s)
        cert = list(self.index_map[1][outliers_in_Yp])
        compl = [self.index_map[1][i] for i in range(self.n) if i not in outliers_in_Yp]
        return cert, compl

    def find_r_cert_and_compl(self, y0):
        indices = np.where(self.bip_adj_mat[:, y0] > 0)[0]
        cert = list(self.index_map[0][indices])
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl


class WeightedClassesPair:
    bip_sim_mat = np.empty((0, 0))
    """The bipartite similarity matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s."""
    bip_adj_mat = np.empty((0, 0))
    """The bipartite adjacency matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s"""
    r = s = -1
    """The classes composing the bipartite graph"""
    n = 0
    """the cardinality of a class"""
    index_map = np.empty((0, 0))
    """A mapping from the bipartite adjacency matrix nodes to the adjacency matrix ones"""
    bip_avg_deg = 0.0
    """the average degree of the graph"""
    bip_density = 0.0
    """the average density of the graph"""
    epsilon = 0.0
    """the epsilon parameter"""

    def __init__(self, sim_mat, adj_mat, classes, r, s, epsilon):
        self.r = r
        self.s = s
        self.index_map = np.where(classes == r)[0]
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        self.bip_sim_mat = sim_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.n = self.bip_sim_mat.shape[0]
        self.bip_avg_deg = self.bip_avg_degree()
        self.bip_density = self.compute_bip_density()
        self.epsilon = epsilon

    def bip_avg_degree(self):
        """
        compute the average degree of the bipartite graph
        :return the average degree
        """
        return (self.bip_sim_mat.sum(0) + self.bip_sim_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        """
        compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the
        bipartite graph
        :return the density
        """
        return self.bip_sim_mat.sum() / (self.n ** 2.0)

    def classes_vertices_degrees(self):
        """
        compute the degree of all vertices in the bipartite graph
        :return a (n,) numpy array containing the degree of each vertex
        """
        c_v_degs = np.sum(self.bip_adj_mat, 0)
        c_v_degs = np.vstack((c_v_degs, np.sum(self.bip_adj_mat, 1)))
        return c_v_degs

    # def neighbourhood_matrix(self, transpose_first=True):
    #     if transpose_first:
    #         return self.bip_adj_mat.T @ self.bip_adj_mat
    #     else:
    #         return self.bip_adj_mat @ self.bip_adj_mat.T
    #
    # def neighbourhood_deviation_matrix(self, nh_mat):
    #     return nh_mat - ((self.bip_avg_deg ** 2.0) / self.n)

    def neighbourhood_deviation_matrix(self, transpose_first=True):
        if transpose_first:
            mat = self.bip_adj_mat.T @ self.bip_adj_mat
        else:
            mat = self.bip_adj_mat @ self.bip_adj_mat.T
        rs_degrees = np.diag(mat)
        mat -= (self.bip_avg_deg ** 2.0) / self.n
        return mat, rs_degrees

    def find_Y(self, nh_dev_mat):
        inner_sums = nh_dev_mat.sum(1) - np.diag(nh_dev_mat)
        inner_sums_indices = np.argsort(inner_sums)[::-1]
        y_card_thresh = int((self.epsilon * self.n) + 1)
        outer_sum = inner_sums[inner_sums_indices[0:(y_card_thresh - 1)]].sum()

        for i in range(y_card_thresh, self.n):
            outer_sum += inner_sums[inner_sums_indices[i]]
            sigma_y = outer_sum / (i ** 2.0)
            if sigma_y >= ((self.epsilon ** 3.0) / 2.0) * self.n:
                return inner_sums_indices[0:i]
        return np.array([])

    def find_Yp(self, degrees, Y_indices):
        return Y_indices[np.abs(degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)]

    def compute_y0(self, nh_dev_mat, Y_indices, Yp_indices):
        sums = np.full((self.n,), -np.inf)
        for i in Yp_indices:
            sums[i] = 0
            for j in list(set(Y_indices) - set(Yp_indices)):
                sums[i] += nh_dev_mat[i, j]
        return np.argmax(sums)

    def find_s_cert_and_compl(self, nh_dev_mat, y0, Yp_indices):
        outliers_in_s = set(np.where(nh_dev_mat[y0, :] > 2.0 * (self.epsilon ** 4.0) * self.n)[0])
        outliers_in_Yp = list(set(Yp_indices) & outliers_in_s)
        cert = list(self.index_map[1][outliers_in_Yp])
        compl = [self.index_map[1][i] for i in range(self.n) if i not in outliers_in_Yp]
        return cert, compl

    def find_r_cert_and_compl(self, y0):
        indices = np.where(self.bip_adj_mat[:, y0] > 0.0)[0]
        cert = list(self.index_map[0][indices])
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl
