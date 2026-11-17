import numpy as np
def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
    """
    Compute learning rate schedule with linear warmup and cosine decay.
    
    Args:
        T: Total number of training steps
        W: Number of warmup steps
        lr_max: Maximum learning rate (reached after warmup)
        lr_min: Minimum learning rate (reached at end of training)
    
    Returns:
        List of learning rates for each step
    """
    # Your code here
    lrs = []
    for i in range(T):
        if i < W:
            lr = lr_max * (i / W)
        else:
            progress = (i - W) / (T - W)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
        lrs.append(float(lr))      
    return lrs