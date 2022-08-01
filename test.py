import numpy as np
from scipy import sparse
A = np.array([[1,2,0],[0,0,3],[1,0,4]])
B = np.matrix([[1,2,0],[0,0,3],[1,0,4]])
A = sparse.csr_matrix(A)

import dgl
graph_dgl = dgl.to_homogeneous(dgl.from_scipy(A))
print(graph_dgl)