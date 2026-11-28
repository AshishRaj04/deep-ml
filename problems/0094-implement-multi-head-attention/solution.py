import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q=X@W_q
    K=X@W_k
    V=X@W_v
    return (Q,K,V)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    dk = Q.shape[-1]
    scores = (Q@K.T)/np.sqrt(dk)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    wei = np.exp(scores)
    wei = wei/np.sum(wei,axis=-1,keepdims=True)
    return wei@V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    dk = Q.shape[-1] // n_heads
    head_output = []
    for h in range(n_heads):
        start = h * dk
        end = (h+1) * dk
        head_out = self_attention(Q[ : ,start:end],K[ : ,start:end],V[ : ,start:end])
        head_output.append(head_out)
    return np.concatenate(head_output,axis=-1)