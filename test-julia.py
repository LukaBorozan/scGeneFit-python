# python3.7
from scGeneFit.functions import *
import numpy as np
import scanpy as sc
np.random.seed(0)
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
def performance(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

# You will have to finish the implementation of this function.
def get_data(filename):
    data = [] # 2D array, each subarray contains the expression of a single cell.
    labels = [] # Label with the index i corresponds to the i-th cell in data.
    return data, labels

# Load data here. Each line in the file is the gene expression of a single cell.
data, labels = get_data("CITEseq_subsample.txt")

# We will be selecting a total of 'num_markers' marker genes. Change it as you like.
num_markers = 25

# How many randomly selected constraints will scGeneFit consider. If you select
# the experimental option when running the solver, this value will not be used.
num_constraints = 1000

# Influences the generation of delta. Best to leave it as it is.
eps = 0.1

# Set solver to 'gurobi' for the improved scGeneFit implementation or 'experimental' for my own implementation.
markers = get_markers(data, labels, num_markers, method = 'pairwise', redundancy = 0.25, solver = 'gurobi', max_constraints = num_constraints, epsilon = eps)

# The code below evaluates how good the markers are.
accuracy=performance(data, labels, data, labels, clf)
accuracy_markers=performance(data[:,markers], labels, data[:,markers], labels, clf)

print("Accuracy (whole data,", data.shape[1], "markers):", accuracy)
print("Accuracy (selected", num_markers, "markers)", accuracy_markers)

markers.sort()
print(markers)
