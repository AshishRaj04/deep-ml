import numpy as np

def bellman_expectation_value(P, R, policy, gamma):
    """
    Compute the state-value function V^pi for a given policy using
    the Bellman expectation equation.
    
    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states)
        R: Rewards, shape (num_states, num_actions, num_states)
        policy: Stochastic policy, shape (num_states, num_actions)
        gamma: Discount factor
    
    Returns:
        State-value function as numpy array of shape (num_states,)
    """
    s, a, s_ = P.shape
    V = np.zeros(s)

    while True:
        V_new = np.zeros(s)

        for state in range(s):
            for action in range(a):
                V_new[state] += policy[state, action] * np.sum(
                    P[state, action] * 
                    (R[state, action] + gamma * V)
                )

        if np.max(np.abs(V_new - V)) < 1e-10:
            break

        V = V_new

    return V_new