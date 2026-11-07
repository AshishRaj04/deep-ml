import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    AT = np.transpose(A)
    ATA = np.matmul(AT, A)
    AAT = np.matmul(A, AT)

    evals_V, V = np.linalg.eig(ATA)

    idx = evals_V.argsort()[::-1]
    evals_V = evals_V[idx]
    V = V[:, idx]

    s = np.sqrt(np.maximum(evals_V, 0))

    U = np.zeros_like(A, dtype=float)
    for i in range(len(s)):
        if s[i] > 1e-10:
            U[ :, i] = np.matmul(A, V[:, i]) / s[i]
        else:
            pass
    return (U, s, V.T)