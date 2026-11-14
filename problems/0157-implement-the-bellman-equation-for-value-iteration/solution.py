import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    # TODO: Implement Bellman update
    V_new = np.zeros_like(V, dtype=float)
    best_action = 0.0
    for v in range(len(V)):
        action_values = []
        for i in transitions[v].values():
            summ = 0.0
            for tr in i:
                prob, next_state, reward, is_done = tr
                summ += prob * (reward + gamma * (0 if is_done else V[next_state]))
            action_values.append(summ)
            
        V_new[v] = max(action_values)
    return V_new