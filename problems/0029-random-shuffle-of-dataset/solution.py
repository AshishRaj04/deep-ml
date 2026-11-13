import numpy as np

def shuffle_data(X, y, seed=None):
    # Your code here
    if seed is not None:
        np.random.seed(seed)
        i = np.arange(X.shape[0])
        np.random.shuffle(i)
        # print(i)
        return X[i] , y[i]
    