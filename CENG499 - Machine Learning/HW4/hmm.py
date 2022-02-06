import numpy as np

def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    N = np.shape(B)[0]
    T = np.shape(O)[0]
    Trellis = np.zeros((N,T))
    for s in range(N):
        Trellis[s][0] = pi[s] * B[s][O[0]]
    for t in range(1,T):
        for s in range(N):
            temp = 0
            for i in range(N):
                temp += Trellis[i][t-1] * A[i][s] * B[s][O[t]]
            Trellis[s][t] = temp
    probability = 0
    for i in range(N):
        probability += Trellis[i,T-1]
    return probability, Trellis

def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    N = np.shape(B)[0]
    T = np.shape(O)[0]
    Trellis = np.zeros((N,T))
    Back_Holder = np.zeros((N,T)).astype(int)
    for s in range(N):
        Trellis[s][0] = pi[s] * B[s][O[0]]
        Back_Holder[s][0] = 0
    for t in range(1,T):
        for s in range(N):
            temp = Trellis[:,t-1] * A[:,s] * B[s][O[t]]
            Trellis[s][t] = np.max(temp)
            Back_Holder[s][t] = np.argmax(temp)
    best = np.argmax(Trellis[:,T-1])
    path = [best]
    t = T-1
    while t > 0:
        path.append(Back_Holder[best][t])
        best = path[-1]
        t -= 1 
    return np.array(path[::-1]), Trellis