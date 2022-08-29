import numpy as np

from scipy.stats import multinomial
from typing import List

def equilibrium_distribution(p_transition):

    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )
    b = np.transpose(np.array([0] * n_states + [1]))
    p_eq = np.linalg.solve(
        a=np.transpose(A).dot(A),
        b=np.transpose(A).dot(b)
    )
    return p_eq

def markov_sequence(p_init, p_transition, sequence_length):
    
    if p_init is None:
        p_init = equilibrium_distribution(p_transition)
        
    initial_state = list(multinomial.rvs(1, p_init)).index(1)

    states = [initial_state]
    for _ in range(sequence_length - 1):
        p_tr = p_transition[states[-1]]
        new_state = list(multinomial.rvs(1, p_tr)).index(1)
        states.append(new_state)
    return states

def get_obs(states, mus, sigmas):

    emissions = []

    for s in states:

        val = np.random.normal(mus[s], sigmas[s], 1)
        emissions.append(val[0])

    return emissions

def get_seq(p, mus, sigmas, length = 1000):

    state_seq = markov_sequence(None, p, length)
    obs = get_obs(state_seq, mus, sigmas)

    true_mu_seq = []

    for s in state_seq:

        true_mu_seq.append(mus[s])

    return obs, true_mu_seq