import numpy as np

def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1
    pos_emb = np.zeros((position,d_model))
    
    for p in range(position):
        for i in range(d_model//2):
            deno = 1 / 10000 ** (2 * i / d_model)
            
            sin_freq = np.sin(p * deno)
            pos_emb[p,2*i] = sin_freq

            cos_freq = np.cos(p * deno)
            pos_emb[p,2*i + 1] = cos_freq
      
    return pos_emb.astype(np.float16)