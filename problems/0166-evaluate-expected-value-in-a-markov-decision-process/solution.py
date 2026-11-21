import numpy as np

def expected_action_value(state, action, P, R, V, gamma):
    """
    Computes the expected value of taking `action` in `state` for the given MDP.
    Args:
      state: int or str, the current state
      action: str, the chosen action
      P: dict of dicts, P[s][a][s'] = prob of next state s' if a in s
      R: dict of dicts, R[s][a][s'] = reward for (s, a, s')
      V: np.ndarray, the value function vector, indexed by state
      gamma: float, discount factor
    Returns:
      float: expected value
    """
    ps, rs = P[state][action].values() , R[state][action].values()
    counter = 0
    Q_s_a = 0
    for p, r in zip(ps, rs):
        Q_s_a += p * (r + gamma * V[counter])
        counter += 1
    return Q_s_a