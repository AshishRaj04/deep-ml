import numpy as np

def to_categorical(x, n_col=None):
    # Your code here
    if n_col == None:
        n_col = max(x) + 1
    out = np.zeros(shape=(len(x), n_col))
    for i in range(len(x)):
        out[i][x[i]] = 1
    return out
