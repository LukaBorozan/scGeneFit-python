# Python3.7
from scGeneFit.newcopyjfunctions import *
import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np
# Testing with random, but the same numbers all the time. random.seed(0) makes the function predictable?
np.random.seed(0)
from sklearn.neighbors import NearestCentroid
import sys
import pdb
import ast

args = sys.argv
test = args
n_cells = args[1]
n_markers = int(args[2])
fixed_genes = args[3]
fixed_genes = ast.literal_eval(fixed_genes)
LPvsILP = args[4]

print(n_cells)
print(n_markers)
print(fixed_genes)
print(LPvsILP)

#pdb.set_trace()

# Method used for clustering: nearest centroid
clf = NearestCentroid()
def performance(X_train, y_train, X_test, y_test, clf):
    # call fit() to train the model using the input training and data
    clf.fit(X_train, y_train)
    # score() determines model accuracy
    return clf.score(X_test, y_test)
# Finish implementation of this function
def get_data(n_cells):
    df = pd.read_table(f"CITEseq_subsample_data{n_cells}.txt", delim_whitespace=True, header=None)
    data = df.to_numpy(dtype=None, copy=False)
    # data = aha # 2D array, each subarray contains the expression of a single cell
    data = np.array(data)
    df_l = pd.read_table(f"CITEseq_subsample_labels{n_cells}.txt", delim_whitespace=True, header=None)
    # labels = sum([[i] * 10 for i in range (13)], []) # Label with the index i corresponds to the i-th cell in data
    labels = df_l[1].to_list()
    # data = aha # 2D array, each subarray contains the expression of a single cell
    # labels = np.array(labels)
    return data, labels
# Each line in the file is the gene expression of a single cell
data, labels = get_data(n_cells)
# Selecting a total of 'num_markers' marker genes
num_markers = n_markers
# Randomly selected constraints that scGeneFit will consider, in the experimental solver, this value is not used
num_constraints = 1000000
# Influences the generation of delta. Best to leave it as it is.
eps = 0.1
# Set solver to 'gurobi' for the improved scGeneFit implementation or 'experimental' for my own implementation.
markers = get_markers(data, labels, num_markers, method = 'pairwise', redundancy = 0.25, solver = 'gurobi', max_constraints = num_constraints, epsilon = eps, fixed_genes = fixed_genes, LPvsILP = LPvsILP)
accuracy=performance(data, labels, data, labels, clf)
accuracy_markers=performance(data[:,markers], labels, data[:,markers], labels, clf)
print("Accuracy (whole data,", data.shape[1], "markers):", accuracy)
print("Accuracy (selected", num_markers, "markers)", accuracy_markers)
# The sort() method sorts the list ascending by default, sort(reverse=True) for sorting of high to low
markers.sort()
#print(markers)
