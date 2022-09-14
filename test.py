from scGeneFit.functions import *

import numpy as np


np.random.seed(0) 

from sklearn.neighbors import NearestCentroid
clf=NearestCentroid()

def performance(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

# 13 cell types
def testfunc(solver, num, fixed_genes = {}):
        #load data from files
        [data, labels, names]= load_example_data("CITEseq")
        #[data, labels, names]= load_example_data("zeisel")
        N,d=data.shape

        num_markers=25
        method='pairwise'
        #method='centers'
        redundancy=0.25
        
        #samples, samples_labels, idx = sample(data, labels, 0.25)
        #eps = optimize_epsilon(np.array(samples), np.array(samples_labels), np.array(data), np.array(labels), num_markers=num_markers, method=method)
        #print(eps)       

        eps = 1.14450351
        markers = get_markers(data, labels, num_markers, method=method, redundancy=redundancy, solver=solver, max_constraints=num, epsilon=eps, fixed_genes=fixed_genes)

        #accuracy=performance(data, labels, data, labels, clf)
        #accuracy_markers=performance(data[:,markers], labels, data[:,markers], labels, clf)

        #print("Accuracy (whole data,", d, " markers): ", accuracy)
        #print("Accuracy (selected", num_markers, "markers)", accuracy_markers)
        
        return markers

def sample(data, labels, sampling_rate):
        """subsample data"""
        indices = []
        for i in set(labels):
                idxs = [x for x in range(len(labels)) if labels[x] == i]
                n = len(idxs)
                s = int(np.ceil(len(idxs) * sampling_rate))
                aux = np.random.permutation(n)[0:s]
                indices += [idxs[x] for x in aux]
        return [data[i] for i in indices], [labels[i] for i in indices], indices

for i in [300000]:
        fixed_genes = {0 : 0, 7 : 1, 4 : 0}
        # markers1 = testfunc('gurobi', i, fixed_genes)
        markers1 = testfunc('experimental', i)
        markers1.sort()
        print(markers1)

# [7, 55, 94, 101, 104, 113, 123, 125, 127, 138, 144, 149, 157, 162, 167, 178, 179, 183, 188, 208, 211, 215, 237, 270, 345]
# [7, 55, 94, 101, 104, 113, 123, 125, 127, 138, 144, 149, 157, 162, 167, 178, 179, 183, 188, 208, 211, 215, 237, 270, 345]

