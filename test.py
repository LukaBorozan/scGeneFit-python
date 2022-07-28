from scGeneFit.functions import *

import numpy as np


np.random.seed(0) 

from sklearn.neighbors import NearestCentroid
clf=NearestCentroid()

def performance(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

# 13 cell types
def testfunc(solver, num):
	#load data from files
	[data, labels, names]= load_example_data("CITEseq")
	#[data, labels, names]= load_example_data("zeisel")
	N,d=data.shape

	num_markers=25
	method='pairwise'
	#method='centers'
	redundancy=0.25
	
	markers = get_markers(data, labels, num_markers, method=method, redundancy=redundancy, solver=solver, max_constraints = num, epsilon=1)

	#accuracy=performance(data, labels, data, labels, clf)
	#accuracy_markers=performance(data[:,markers], labels, data[:,markers], labels, clf)

	#print("Accuracy (whole data,", d, " markers): ", accuracy)
	#print("Accuracy (selected", num_markers, "markers)", accuracy_markers)
	
	return markers

for i in [34073]:
	print(i)
	markers1 = testfunc('experimental', i)
	markers1.sort()
	print(markers1)

# data[:870]

# experimental
# constraints to ~optimum - 35028, / 229241 constraints, 18 constraints left (~0.008%)

# gurobi/scipy
# for 34073 another 33760 are violated
