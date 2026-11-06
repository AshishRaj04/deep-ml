import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)
    exp_score = [math.exp(z - max_score) for z in scores]
    s = sum(exp_score)
    return [e / s for e in exp_score]