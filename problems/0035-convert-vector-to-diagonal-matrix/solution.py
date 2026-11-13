import numpy as np

def make_diagonal(x):
    # Your code here
    n = len(x)
    dtype = x.dtype
    diag_mat = np.zeros((n,n), dtype=dtype)
    for i in range(n):
        diag_mat[i][i] = x[i]
    return diag_mat
    