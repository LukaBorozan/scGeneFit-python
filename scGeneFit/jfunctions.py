import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import sklearn
import sklearn.manifold
import scipy.io
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import scipy.sparse as sp
from . import data_files
from annoy import AnnoyIndex
import sys
import math

def get_markers(data, labels, num_markers, method='centers', epsilon=1, sampling_rate=1, n_neighbors=3, max_constraints=1000, redundancy=0.01, verbose=True, solver='gurobi', fixed_genes = {}, LPvsILP = "continuous"):
    """marker selection algorithm
    data: Nxd numpy array with point coordinates, N: number of points, d: dimension
    labels: list with labels (N labels, one per point)
    num_markers: target number of markers to select. num_markers<d
    method: 'centers', 'pairwise', or 'pairwise_centers'
    epsilon: constraints will be of the form expr>Delta, where Delta is chosen to be epsilon times the norm of the smallest constraint (default 1)
    (This is the most important parameter in this problem, it determines the scale of the constraints, 
    the rest the rest of the parameters only determine the size of the LP)
    sampling_rate: (if method=='pairwise' or 'pairwise_centers') selects constraints from a random sample of proportion sampling_rate (default 1)
    n_neighbors: (if method=='pairwise') chooses the constraints from n_neighbors nearest neighbors (default 3)
    max_constraints: maximum number of constraints to consider (default 1000)
    redundancy: (if method=='centers') in this case not all pairwise constraints are considered 
    but just between centers of consecutive labels plus a random fraction of constraints given by redundancy
    if redundancy==1 all constraints between pairs of centers are considered """
    
    x = []
    
    if (solver != 'experimental'):
        d = data.shape[1]
    
        samples, samples_labels, idx = __sample(data, labels, sampling_rate)

        if method == 'pairwise_centers':
            constraints, smallest_norm = __select_constraints_centers(
                data, labels, samples, samples_labels)
        elif method == 'pairwise':
            constraints, smallest_norm = __select_constraints_pairwise(
                data, labels, samples, samples_labels, n_neighbors)
        else:
            constraints, smallest_norm = __select_constraints_summarized(data, labels, redundancy)

        num_cons = constraints.shape[0]
    
        if num_cons > max_constraints:
            p = np.random.permutation(num_cons)[0:max_constraints]
            constraints = constraints[p, :]
        
        if verbose: 
            print('Solving a linear program with {} variables and {} constraints'.format(
                constraints.shape[1], constraints.shape[0]))

        t = time.time()

        if solver == 'gurobi':
            x, y = __lp_markers_gurobi(constraints, num_markers, smallest_norm * epsilon, fixed_genes, LPvsILP) 
        elif solver == 'scipy':
            sol = __lp_markers(constraints, num_markers, smallest_norm * epsilon)
            x = sol['x'][0:d]
        else:
    	    print("ERROR, valid solvers are gurobi and scipy.")    
    	    return []
    else:
    	t = time.time()
    	x = __lp_markers_cutting(data, labels, num_markers, epsilon, fixed_genes, verbose) 
      
    if verbose:
        print('Time elapsed: {} seconds'.format(time.time() - t))
        
    markers = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[: num_markers]

    return markers


def get_markers_hierarchy(data, labels, num_markers, method='centers', sampling_rate=0.1, n_neighbors=3, epsilon=10, max_constraints=1000, redundancy=0.01, verbose=True):
    """marker selection algorithm with hierarchical labels
    data: Nxd numpy array with point coordinates, N: number of points, d: dimension
    labels: list with T lists of labels, where T is the number of layers in the hierarchy (N labels per list, one per point)
    num_markers: target number of markers to select. num_markers<d
    sampling_rate: selects constraints from a random sample of proportion sampling_rate (default 1)
    n_neighbors: chooses the constraints from n_neighbors nearest neighbors (default 3)
    epsilon: Delta is chosen to be epsilon times the norm of the smallest constraint (default 10)
    max_constraints: maximum number of constraints to consider (default 1000)
    method: 'centers', 'pairwise' or 'pairwise_centers' (default 'centers') 
    redundancy: (if method=='centers') in this case not all pairwise constraints are considered 
    but just between centers of consecutive labels plus a random fraction of constraints given by redundancy
    if redundancy==1 all constraints between pairs of centers are considered"""
    t = time.time()
    [N, d] = data.shape
    num_levels = len(labels)
    prev_label = [1 for i in range(N)]
    constraints = None
    smallest_norm = np.inf
    for i in range(num_levels):
        s = set(prev_label)
        for l in s:
            if l is not None:
                aux_data = [data[x, :]
                            for x in range(len(labels[i])) if prev_label[x] == l]
                aux_labels = [labels[i][x]
                              for x in range(len(labels[i])) if prev_label[x] == l]
                samples, samples_labels, idx = __sample(
                    aux_data, aux_labels, sampling_rate)
                aux_data = np.array(aux_data)

                if method == 'pairwise_centers':
                    con, sm_norm = __select_constraints_centers(
                        aux_data, aux_labels, samples, samples_labels)
                elif method == 'pairwise':
                    con, sm_norm = __select_constraints_pairwise(
                        aux_data, aux_labels, samples, samples_labels, n_neighbors)
                else: 
                    con, sm_norm = __select_constraints_summarized(aux_data, aux_labels, redundancy)

                if constraints is not None:
                    constraints = np.concatenate((constraints, con))
                else:
                    constraints = con
                if sm_norm < smallest_norm:
                    smallest_norm = sm_norm
        prev_label = labels[i]
    constraints = np.array(constraints)
    num_cons = constraints.shape[0]
    if num_cons > max_constraints:
        p = np.random.permutation(num_cons)[0:max_constraints]
        constraints = constraints[p, :]
    if verbose:
        print('Solving a linear program with {} variables and {} constraints'.format(constraints.shape[1], constraints.shape[0]))
    sol = __lp_markers(constraints, num_markers, smallest_norm * epsilon)
    if verbose:
        print('Time elapsed: {} seconds'.format(time.time() - t))
    x = sol['x'][0:d]
    markers = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[
        : num_markers]
    return markers


def __sample(data, labels, sampling_rate):
    """subsample data"""
    indices = []
    for i in set(labels):
        idxs = [x for x in range(len(labels)) if labels[x] == i]
        n = len(idxs)
        s = int(np.ceil(len(idxs) * sampling_rate))
        aux = np.random.permutation(n)[0:s]
        indices += [idxs[x] for x in aux]
    return [data[i] for i in indices], [labels[i] for i in indices], indices


def __select_constraints_summarized(data, labels, redundancy=0.01):
    """selects constraints of the form c_a-c_(a+1) where c_i's are the empirical centers of different classes"""
    constraints = []
    centers = {}
    smallest_norm = np.inf
    labels_set = list(set(labels))
    k = len(labels_set)
    for idx in labels_set:
        X = [data[x, :] for x in range(len(labels)) if labels[x] == idx]
        centers[idx] = np.array(X).mean(axis=0)
    for i in range(len(labels_set)):
        v = centers[labels_set[i]]-centers[labels_set[(i+1) % k]]
        constraints += [v]
        if np.linalg.norm(v) ** 2 < smallest_norm:
            smallest_norm = np.linalg.norm(v) ** 2
        for j in range(len(labels_set)):
            if j != i and j != (i+1) % k:
                if np.random.rand() < redundancy:
                    v = centers[labels_set[j]]-centers[labels_set[(j+1) % k]]
                    constraints += [v]
                    if np.linalg.norm(v) ** 2 < smallest_norm:
                        smallest_norm = np.linalg.norm(v) ** 2
    constraints = np.array(constraints)
    return -constraints * constraints, smallest_norm


def __select_constraints_pairwise(data, labels, samples, samples_labels, n_neighbors):
    """select constraints of the form x-y where x,y have different labels"""
    constraints = []
    # nearest neighbors are selected from the entire set
    neighbors = {}
    data_by_label = {}
    smallest_norm = np.inf
    for i in set(labels):
        X = [data[x, :] for x in range(len(labels)) if labels[x] == i]
        data_by_label[i] = X
        neighbors[i] = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors).fit(np.array(X))
    # compute nearest neighbor for samples
    for i in neighbors.keys():
        Y = [samples[x]
             for x in range(len(samples_labels)) if samples_labels[x] == i]
        for j in neighbors.keys():
            if i != j:
                idx = neighbors[j].kneighbors(Y)[1]
                for s in range(len(Y)):
                    for t in idx[s]:
                        v = Y[s] - data_by_label[j][t]
                        constraints += [v]
                        if np.linalg.norm(v) ** 2 < smallest_norm:
                            smallest_norm = np.linalg.norm(v) ** 2
    constraints = np.array(constraints)
    return -constraints * constraints, smallest_norm


def __select_constraints_centers(data, labels, samples, samples_labels):
    """select constraints of the form (x-ct')^2 - (x-ct)^2> Delta^2 y where x belongs to cluster with center ct"""
    constraints = []
    # nearest neighbors are selected from the entire set
    centers_by_label = {}
    smallest_norm = np.inf
    for i in set(labels):
        X = np.array([data[x, :]
                      for x in range(len(labels)) if labels[x] == i])
        centers_by_label[i] = np.sum(X, axis=0) / X.shape[0]
    # compute nearest neighbor for samples
    for p in range(len(samples)):
        # distance to it's own center
        aux0 = (samples[p] - centers_by_label[samples_labels[p]]) * \
            (samples[p] - centers_by_label[samples_labels[p]])
        for i in set(labels):
            if samples_labels[p] != i:
                # distance to other centers
                aux1 = (samples[p] - centers_by_label[i]) * \
                    (samples[p] - centers_by_label[i])
                constraints += [aux0 - aux1]
                if np.linalg.norm(aux0 - aux1) < smallest_norm:
                    smallest_norm = np.linalg.norm(aux0-aux1)
    constraints = np.array(constraints)
    return constraints, smallest_norm


def __lp_markers(constraints, num_markers, epsilon):
    m, d = constraints.shape
    c = np.concatenate((np.zeros(d), np.ones(m)))
    l = np.zeros(d + m)
    u = np.concatenate((np.ones(d), np.array([float('inf') for i in range(m)])))
    aux1 = np.concatenate((constraints, -np.identity(m)), axis=1)
    aux2 = np.concatenate((np.ones((1, d)), np.zeros((1, m))), axis=1)
    A = np.concatenate((aux1, aux2), axis=0)
    b = np.concatenate((-epsilon * np.ones(m), np.array([num_markers])))
    bounds = [(l[i], u[i]) for i in range(d + m)]
    #sol = scipy.optimize.linprog(c, A, b, None, None, bounds, options = {'tol' : 1e-12})
    sol = scipy.optimize.linprog(c, A, b, None, None, bounds)
    return sol
    

# gurobi solver takes constraints, desired number of markers and tolerance,
# returns (alpha, beta) solution to the LP
def __lp_markers_gurobi(constraints, num_markers, epsilon, fixed, LPvsILP, warm = None):
    M = gp.Model("whatever_bro")
    M.setParam('OutputFlag', 0) 
    m, d = constraints.shape
    
    # create vars
    u = np.concatenate((
        np.array([fixed.get(i, 1) for i in range(d)]),
        float('inf') * np.ones(m)
    ))

    l = np.array(
       [fixed.get(i, 0) for i in range(d + m)]
    )
    
    LPvsILPdict = {"continuous": [GRB.CONTINUOUS, u, l], "binary": [GRB.BINARY, 1, 0]}
    vtype, ub, lb = LPvsILPdict[LPvsILP]
    
    x = M.addMVar(shape = d + m, vtype = vtype, ub = ub, lb = lb)
    # x = M.addMVar(shape = d + m, vtype = GRB.CONTINUOUS, ub = u, lb = l)
    
    # warm start if can
    if (not warm is None):
        x.Start = np.concatenate((warm, np.zeros(m)))
    
    # objective function
    c = np.concatenate((
        np.zeros(d), 
        np.ones(m)
    ))
    
    M.setObjective(c @ x, GRB.MINIMIZE)
        
    # system matrix - constraints, betas, alphas
    row = np.concatenate((
        np.repeat(np.arange(m), d),
        np.arange(m), 
        m * np.ones(d)
    ))
    
    col = np.concatenate((
        np.tile(np.arange(d), m),
        np.arange(d, d + m), 
        np.arange(d)
    ))
    
    val = np.concatenate((
        constraints.flatten(), 
        -np.ones(m), 
        np.ones(d)
    ))
    
    A = sp.csr_matrix(
        (val, (row, col)), 
        shape = (m + 1, d + m)
    )
    
    # rhs vector
    b = np.concatenate((
        -epsilon * np.ones(m), 
        np.array([num_markers])
    ))
    
    # add constraints
    M.addConstr(A @ x <= b)
    
    # optimize
    M.optimize()

    # return alphas, betas
    alpha = np.array(x[:d].X)
    pd.set_option('display.max_rows', None)
    alphas = pd.DataFrame({"alpha": alpha})
    nonzero = alphas[alphas['alpha'] > 0]
    nonzerosorted = nonzero.sort_values(by='alpha', ascending=False)
    newdict = nonzerosorted.to_dict('dict')
    zero = alphas[alphas['alpha'] == 0]
    newzero = zero.to_dict('dict')
    print("Length fractional:", len(newdict))
    print("Fractional:", newdict)
    print("Length zeros:", len(newzero))
    print("Zeros:", newzero)
    print("Objective value:", M.ObjVal)
    return np.array(x[:d].X), np.array(x[d:].X)

def __lp_markers_cutting(data, labels, num_markers, epsilon, fixed, verbose):
    # number of genes per cell
    d = data.shape[1]
    
    # used for the removal of the constraints
    D = set([]) # stored pairs of vertices that have been added to the constraints
    offset = 0  # index tracker for the pairs of vertices
    M = {}      # maps indices into pairs
    prev = []   # previous iteration

    # main loop
    for i in range(10000):
        if verbose:
            print("iteration:", i + 1, flush = True)

        # is it the initial run?
        isFirstRun = i == 0

        # subsample data for the initial iteration
        if isFirstRun:
            #samples = data
            #sample_labels = labels
            #idx = None
            samples, sample_labels, idx = __sample(data, labels, 0.05)
            samples = np.array(samples)

        # debug
        tt = time.time()
    
        # building annoy index
        if (isFirstRun):
            x = np.concatenate((
                np.ones(num_markers),
                np.zeros(d - num_markers)
            ))
            # x = np.ones(d)
            I = __get_annoy_index(samples, x, num_markers)
        else:
            # considering indices with biggest alpha values
            I = __get_annoy_index(data, x, num_markers)
        
        if (isFirstRun):
            # in the first run, we build delta as a minimum squared distance of the 
            # nearest neighbors of all cells
            delta = __get_eps(samples, sample_labels, I) * epsilon # 0.05828751115486528
            print("delta:", delta, flush = True)
        #elif i % 5 == 4:
            # TODO: in all other runs, we remove inactive constraints from the system matrix
            #B = __prune_cutting_matrix(B, D, M, x, y, delta)
            #offset = B.shape[0]
            
        # debug
        if verbose:
            print("index:", "{:.2f}".format(time.time() - tt), "sec", flush = True)
        tt = time.time()
        
        # building new constraints here
        if (isFirstRun):
            A = __get_cutting_constraints(samples, sample_labels, x, I, D, M, offset, delta, idx)
        else:
            A = __get_cutting_constraints(data, labels, x, I, D, M, offset, delta, None)
        
        # releasing annoy index
        I.unbuild() 
        
        # if there are no new constraints to add, we consider the iterations successful
        if (A is None):
            break
            
        if (isFirstRun):
            B = A
        else:
            # appending new constraints to the previous ones and updating the offset
            B = np.concatenate((B, A))
        
        offset = B.shape[0]
        
        # debug
        if verbose:
            print("cons:", "{:.2f}".format(time.time() - tt), "sec", flush = True)
        tt = time.time()
        
        # solving the model, -B is passed for legacy reasons
        if (isFirstRun):
            x, y = __lp_markers_gurobi(-B, num_markers, delta, fixed) 
        else:
            x, y = __lp_markers_gurobi(-B, num_markers, delta, fixed, x)
        
        # debug, can delete later
        if verbose:
            print("solving:", B.shape, "{:.2f}".format(time.time() - tt), "sec", flush = True)
        # print(i + 1, "sum_beta:", sum(list(y)), "shape:", B.shape, flush = True)
       
		# current iteration
        if (prev != [] and np.array_equal(prev, x)):
            break
        else:
            prev = x

        zz = [(i, x[i]) for i in range(d) if x[i] > 5e-12] 
        print(len(zz), zz)

        # set to True for per-iteration performance checking
        if False:
            tt = time.time()
            performance_metrics(data, labels, x, delta, num_markers, D)
            print("debugging:", time.time() - tt, "sec")

    # final performance checking
    if False:
        performance_metrics(data, labels, x, delta, num_markers, D)

    return x


# removes inactive constraints, updates B, D and M
def __prune_cutting_matrix(B, D, M, x, y, delta):
    R = range(B.shape[0])

    I = [
        i for i in R
        if np.dot(B[i, :], x) + y[i] >= (1 + 1e-6) * delta
    ]
    
    D = D.difference([M[i] for i in I])
    A = list(set(R).difference(I))
    M = { i : M[A[i]] for i in range(len(A)) }

    # debug
    print("pruned", B.shape[0] - len(A), flush = True)

    return B[A, :]
    

# builds a new annoy index for the selected dimensions
def __get_annoy_index(data, alpha, num_markers):
    #x = [i for i in range(len(alpha)) if alpha[i] > 5e-12]
    x = sorted(
        [i for i in range(len(alpha)) if alpha[i] > 5e-12],
        key = lambda i : alpha[i], 
        reverse = True
    )[:num_markers]

    n = data.shape[0]
    d = len(x)
    a = alpha[x]

    I = AnnoyIndex(d, 'euclidean')
    I.set_seed(0)
    for i in range(n - 1, -1, -1):
        v = data[i, x] 
        I.add_item(i, a * v * v)
    I.build(10, n_jobs = -1)

    return I


# builds a new annoy index for all coordinates
def __get_annoy_index_full_dimension(data, alpha):
    n, d = data.shape
    I = AnnoyIndex(d, 'euclidean')
    I.set_seed(0)
    for i in range(n - 1, -1, -1):
        v = data[i, :]
        v = alpha * v * v
        I.add_item(i, v) 
    I.build(10, n_jobs = -1)
    return I
    

# calculates the smallest nonzero distance between cells with different labels
# and squares it to be used as delta
def __get_eps(data, labels, I):
    return min([
        n
        for j in range(data.shape[0])
        for i in I.get_nns_by_item(j, 20)
        for v in [data[j, :] - data[i, :]]
        for n in [np.dot(v, v)]
        if n > 5e-12 and labels[i] != labels[j]
    ])


# TODO: vectorize? sparsify?
def __get_cutting_constraints(data, labels, alpha, I, D, M, offset, delta, idx):
    isFirstRun = offset == 0

    # number of cells
    n, d = data.shape
    # number of nearest neighbors to iterate over
    nNeighbors = max(int(n / 400) + 1, 20)
    # maximum number of constraints to add per cell
    nConstraints = 1 # if isFirstRun else 10
    # new constraints will be stored here
    A = []

    # introduce constraints to A, but only if they were violated in the previous iteration
    # if its the first iteration
    # considering pairs of cells (i, j) for all cells i and its m nearest neighbors
    for i in range(n):
        nns = [ # only consider nearest neighbors with different labels
            j for j in I.get_nns_by_item(i, nNeighbors) 
            if labels[i] != labels[j]
        ]

        x = data[i, :]
        k = 0

        for j in nns:
            ai = min(i, j) if idx is None else min(idx[i], idx[j])
            aj = max(i, j) if idx is None else max(idx[i], idx[j])

            # do not consider pairs that we have already seen
            if (ai, aj) in D:
                continue

            # a constraint of form alpha * (xi - xj)^2
            v = (x - data[j, :]) ** 2

            if (np.sum(v) < delta):
                continue

            if (np.dot(alpha, v) < delta):
                D.add((ai, aj))
                M[offset] = (ai, aj)
                offset += 1
                A.append(v)
                k += 1
            # nearest neighbors (j's) are sorted, it is safe to break
            else:  
                break
            
            # add nConstraints constraints in every iteration
            if k == nConstraints:
                break

    if (A == []):
        return None
    
    return np.array(A)


def performance_metrics(data, label, alpha, delta, num_markers, D = None):
    n, d = data.shape
    a = 0
    b = 0
    
    for i in range(n):
        x = data[i, :]
        for j in range(i + 1, n):
            if label[i] != label[j]:
                a += 1
                
                if (not D is None and (i, j) in D):
                    continue
                
                v = x - data[j, :]
                if (np.dot(alpha, v * v) < (1 - 1e-6) * delta):
                    b += 1
    
    print("violated:", b, "total:", a, "percentage", b / a)
    


def circles_example(N=30, d=5):
    num_markers = 2
    X = np.concatenate((np.array([[np.sin(2 * np.pi * i / N), np.cos(2 * np.pi * i / N)] for i in range(N)]),
                        np.random.random((N, d - 2))), axis=1)
    Y = np.concatenate((np.array([[2 * np.sin(2 * np.pi * i / N), 2 * np.cos(2 * np.pi * i / N)] for i in range(N)]),
                        np.random.random((N, d - 2))), axis=1)
    data = np.concatenate((X, Y), axis=0)
    labels = np.concatenate((np.zeros(10), np.ones(10)))
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[0:N, 0], data[0:N, 1], data[0:N, 2], c='r', marker='o')
    ax.scatter(data[N + 1:2 * N, 0], data[N + 1:2 * N, 1],
               data[N + 1:2 * N, 2], c='g', marker='x')
    plt.show()
    sol = get_markers(data, labels, num_markers, 1, 3, 10)
    x = sol['x'][0:d]
    markers = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[
        :num_markers]
    for i in range(d):
        if i not in markers:
            data[:, i] = np.zeros(2 * N)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(data[0:N, 0], data[0:N, 1], data[0:N, 2], c='r', marker='o')
    ax2.scatter(data[N + 1:2 * N, 0], data[N + 1:2 * N, 1],
                data[N + 1:2 * N, 2], c='g', marker='x')
    plt.show()


def plot_marker_selection(data, markers, names, perplexity=40):
    print('Computing TSNE embedding')
    t = time.time()
    X_original = sklearn.manifold.TSNE(
        n_components=2, perplexity=perplexity).fit_transform(data)
    X_embedded = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity).fit_transform(
        data[:, markers])
    print('Elapsed time: {} seconds'.format(time.time() - t))
    cmap = plt.cm.jet
    unique_names = list(set(names))
    num_labels = len(unique_names)
    colors = [cmap(int(i * 256 / num_labels)) for i in range(num_labels)]
    aux = [colors[unique_names.index(name)] for name in names]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    for g in unique_names:
        i = [s for s in range(len(names)) if names[s] == g]
        ax.scatter(X_original[i, 0], X_original[i, 1],
                   c=[aux[i[0]]], s=5, label=names[i[0]])
    ax.set_title('Original data')
    ax2 = fig.add_subplot(122)
    for g in np.unique(names):
        i = [s for s in range(len(names)) if names[s] == g]
        ax2.scatter(X_embedded[i, 0], X_embedded[i, 1],
                    c=[aux[i[0]]], s=5, label=names[i[0]])
    ax2.set_title('{} markers'.format(len(markers)))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.7)
    return fig


def one_vs_all_selection(data, labels, num_bins=20):
    data_by_label = {}
    unique_labels = list(set(labels))
    number_classes = len(unique_labels)
    [N, d] = data.shape
    for lab in unique_labels:
        X = [data[x, :] for x in range(len(labels)) if labels[x] == lab]
        data_by_label[lab] = X
    markers = [None for i in range(number_classes)]
    bins = data.max() / num_bins * range(num_bins + 1)
    for idx in range(number_classes):
        c = unique_labels[idx]
        current_class = np.array(data_by_label[c])
        others = np.concatenate([data_by_label[lab]
                                 for lab in unique_labels if lab != c])
        big_dist = 0
        for gene in range(d):
            if gene not in markers[0:idx]:
                [h1, b1] = np.histogram(current_class[:, gene], bins)
                h1 = np.array(h1).reshape(1, -1) / current_class.shape[0]
                [h2, b2] = np.histogram(others[:, gene], bins)
                h2 = np.array(h2).reshape(1, -1) / others.shape[0]
                dist = -sklearn.metrics.pairwise.additive_chi2_kernel(h1, h2)
                if dist > big_dist:
                    markers[idx] = gene
                    big_dist = dist
    return markers


def optimize_epsilon(data_train, labels_train, data_test, labels_test, num_markers, method='centers', solver='experimental', fixed_parameters={}, bounds=[(0.2 , 10)], x0=[1], max_fun_evaluations=20, n_experiments=5, clf=None, hierarchy=False, verbose=True):
    """
    Finds the optimal value of epsilon using scipy.optimize.dual_annealing
    """
    if clf==None:
        clf=sklearn.neighbors.NearestCentroid()
    Instance=__ScGeneInstance(data_train, labels_train, data_test, labels_test, clf, num_markers, method, fixed_parameters, n_experiments, hierarchy, solver)
    print('Optimizing epsilon for', num_markers, 'markers and', method, 'method.')
    res = scipy.optimize.dual_annealing(Instance.error_epsilon, bounds=bounds, x0=x0,  maxfun=max_fun_evaluations, no_local_search=True)
    return [res.x, 1-res.fun]    

class __ScGeneInstance:
    def __init__(self, X_train, y_train, X_test, y_test, clf, num_markers, method, fixed_parameters, n_experiments, hierarchy, solver):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.clf=clf
        self.num_markers=num_markers
        self.method=method
        self.fixed_parameters=fixed_parameters
        self.n_experiments=n_experiments
        self.hierarchy=hierarchy
        self.solver=solver
    def error_epsilon(self, epsilon):
        return 1-self.accuracy(epsilon)

    def accuracy(self, epsilon):
        #compute avg over n_experiments random samples for stability
        if self.hierarchy:
            markers=[get_markers_hierarchy(self.X_train, self.y_train, self.num_markers, self.method, epsilon=epsilon, verbose=False, **self.fixed_parameters) for i in range(self.n_experiments)]
        else:    
            markers=[get_markers(self.X_train, self.y_train, self.num_markers, self.method, solver=self.solver, epsilon=epsilon, verbose=False, **self.fixed_parameters) for i in range(self.n_experiments)]
        val=[self.performance( markers[i] ) for i in range(self.n_experiments)]
        return np.mean(val)
    
    def performance(self, markers):
        if self.hierarchy:
            self.clf.fit(self.X_train[:,markers], self.y_train[0])
            return self.clf.score(self.X_test[:,markers], self.y_test[0])
        else:
            self.clf.fit(self.X_train[:,markers], self.y_train)
            return self.clf.score(self.X_test[:,markers], self.y_test)

def load_example_data(name):
    if name=="CITEseq":
        a = scipy.io.loadmat(data_files.get_data("CITEseq.mat"))
        data= a['G'].T
        N,d=data.shape
        #transformation from integer entries 
        data=np.log(data+np.ones(data.shape))
        for i in range(N):
            data[i,:]=data[i,:]/np.linalg.norm(data[i,:])
        #load labels from file
        a = scipy.io.loadmat(data_files.get_data("CITEseq-labels.mat"))
        l_aux = a['labels']
        labels = np.array([i for [i] in l_aux])
        #load names from file
        a = scipy.io.loadmat(data_files.get_data("CITEseq_names.mat"))
        names=[a['citeseq_names'][i][0][0] for i in range(N)]
        return [data, labels, names]
    elif name=="zeisel":
        #load data from file
        a = scipy.io.loadmat(data_files.get_data("zeisel_data.mat"))
        data= a['zeisel_data'].T
        N,d=data.shape

        #load labels (first level of the hierarchy) from file
        a = scipy.io.loadmat(data_files.get_data("zeisel_labels1.mat"))
        l_aux = a['zeisel_labels1']
        l_0=[l_aux[i][0] for i in range(l_aux.shape[0])]
        #load labels (second level of the hierarchy) from file
        a = scipy.io.loadmat(data_files.get_data("zeisel_labels2.mat"))
        l_aux = a['zeisel_labels2']
        l_1=[l_aux[i][0] for i in range(l_aux.shape[0])]
        #construct an array with hierarchy labels
        labels=np.array([l_0, l_1])

        # load names from file 
        a = scipy.io.loadmat(data_files.get_data("zeisel_names.mat"))
        names0=[a['zeisel_names'][i][0][0] for i in range(N)]
        names1=[a['zeisel_names'][i][1][0] for i in range(N)]
        return [data, labels, [names0,names1]]
    else:
        print("currently available options are only 'CITEseq' and 'zeisel'")
