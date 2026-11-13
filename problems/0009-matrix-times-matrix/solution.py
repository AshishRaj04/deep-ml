import numpy as np
def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
    a = np.array(a)
    b = np.array(b)
    if a.shape[1] != b.shape[0]:
        return -1
    else:
        m,n = a.shape[0], b.shape[1]
        c = np.zeros((m,n), dtype=a.dtype)
        for i in range(m):
            for j in range(n):
                c[i][j] = np.dot(a[i] , b[ : , j])
        return c
            