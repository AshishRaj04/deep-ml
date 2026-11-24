import numpy as np

def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack even-length nibble array (0..15) into uint8 bytes: lo | (hi<<4)."""
    # Your code here
    out = []
    for i in range(0,len(q),2):
        lo, hi = q[i:i+2]
        out.append(lo | (hi<<4))
    return np.array(out, dtype=np.uint8)
    

def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack bytes to length-n int nibbles."""
    # You can think of & 0x0F as a filter that keeps the lower half, while >> 4 moves the upper half down
    out = []
    for byte in packed:
        lo = byte & 0x0F # 0x0F is hexadecimal notation for 15
        hi = byte >> 4
        out.append(lo)
        out.append(hi)
    return np.array(out[:n], dtype=np.int8)
    

def w4a16_matmul(x, packed_w, scales, in_features: int):
    """
    x: (batch, in_features) float
    packed_w: (out_features, in_features//2) uint8
    scales: (out_features,)
    Dequant: W = scale * (q - 8), y = x @ W.T
    """
    x = np.asarray(x, dtype=np.float32)

    W = []

    for i in range(packed_w.shape[0]):
        q_W = unpack_int4(packed_w[i], in_features).astype(np.float32)
        w = scales[i] * (q_W - 8)
        W.append(w)

    W = np.asarray(W, dtype=np.float32)

    return x @ W.T
