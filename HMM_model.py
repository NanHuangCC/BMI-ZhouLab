"""
This file for HMM study
"""
import numpy as np


def simulate(T):

    def draw_from(probs):
        """
        1.np.random.multinomial:
        按照多项式分布，生成数据
        >>> np.random.multinomial(20, [1/6.]*6, size=2)
                array([[3, 4, 3, 3, 4, 3],
                       [2, 4, 3, 4, 0, 7]])
         For the first run, we threw 3 times 1, 4 times 2, etc.
         For the second, we threw 2 times 1, 4 times 2, etc.
        2.np.where:
        >>> x = np.arange(9.).reshape(3, 3)
        >>> np.where( x > 5 )
        (array([2, 2, 2]), array([0, 1, 2]))
        """
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]

    observations = np.zeros(T, dtype=int)
    states = np.zeros(T, dtype=int)
    states[0] = draw_from(pi)
    observations[0] = draw_from(B[states[0],:])
    for t in range(1, T):
        states[t] = draw_from(A[states[t-1],:])
        observations[t] = draw_from(B[states[t],:])
    return observations, states


def generate_index_map(lables):
    id2label = {}
    label2id = {}
    i = 0
    for l in lables:
        id2label[i] = l
        label2id[l] = i
        i += 1
    return id2label, label2id


def convert_map_to_vector(map_, label2id):
    """将概率向量从dict转换成一维array"""
    v = np.zeros(len(map_), dtype=float)
    for e in map_:
        v[label2id[e]] = map_[e]
    return v


def convert_map_to_matrix(map_, label2id1, label2id2):
    """将概率转移矩阵从dict转换成矩阵"""
    m = np.zeros((len(label2id1), len(label2id2)), dtype=float)
    for line in map_:
        for col in map_[line]:
            m[label2id1[line]][label2id2[col]] = map_[line][col]
    return m


def forward(obs_seq):
    """前向算法"""
    N = A.shape[0]
    T = len(obs_seq)

    # F保存前向概率矩阵
    F = np.zeros((N, T))
    F[:, 0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for n in range(N):
            F[n, t] = np.dot(F[:, t - 1], (A[:, n])) * B[n, obs_seq[t]]

    return F


def backward(obs_seq):
    """后向算法"""
    N = A.shape[0]
    T = len(obs_seq)
    # X保存后向概率矩阵
    X = np.zeros((N,T))
    X[:,-1:] = 1

    for t in reversed(range(T-1)):
        for n in range(N):
            X[n,t] = np.sum(X[:,t+1] * A[n,:] * B[:, obs_seq[t+1]])

    return X


def baum_welch_train(observations, A, B, pi, criterion=0.05):
    """无监督学习算法——Baum-Weich算法"""
    n_states = A.shape[0]
    n_samples = len(observations)

    done = False
    while not done:
        # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
        # Initialize alpha
        alpha = forward(observations)

        # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
        # Initialize beta
        beta = backward(observations)
        # ξ_t(i,j)=P(i_t=q_i,i_{i+1}=q_j|O,λ)
        xi = np.zeros((n_states, n_states, n_samples - 1))
        for t in range(n_samples - 1):
            denom = np.dot(np.dot(alpha[:, t].T, A) * B[:, observations[t + 1]].T, beta[:, t + 1])
            for i in range(n_states):
                numer = alpha[i, t] * A[i, :] * B[:, observations[t + 1]].T * beta[:, t + 1].T
                xi[i, :, t] = numer / denom

        # γ_t(i)：gamma_t(i) = P(q_t = S_i | O, hmm)
        gamma = np.sum(xi, axis=1)
        # Need final gamma element for new B
        # xi的第三维长度n_samples-1，少一个，所以gamma要计算最后一个
        prod = (alpha[:, n_samples - 1] * beta[:, n_samples - 1]).reshape((-1, 1))
        gamma = np.hstack((gamma, prod / np.sum(prod)))  # append one more to gamma!!!

        # 更新模型参数
        newpi = gamma[:, 0]
        newA = np.sum(xi, 2) / np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))
        newB = np.copy(B)
        num_levels = B.shape[1]
        sumgamma = np.sum(gamma, axis=1)
        for lev in range(num_levels):
            mask = observations == lev
            newB[:, lev] = np.sum(gamma[:, mask], axis=1) / sumgamma

        # 检查是否满足阈值
        if np.max(abs(pi - newpi)) < criterion and \
                np.max(abs(A - newA)) < criterion and \
                np.max(abs(B - newB)) < criterion:
            done = 1
        A[:], B[:], pi[:] = newA, newB, newpi
    return newA, newB, newpi


def viterbi(obs_seq, A, B, pi):
    """
    Returns
    -------
    V : nu mpy.ndarray
        V [s][t] = Maximum probability of an observation sequence ending
                   at time 't' with final state 's'
    prev : numpy.ndarray
        Contains a pointer to the previous state at t-1 that maximizes
        V[state][t]

    V对应δ，prev对应ψ
    """
    N = A.shape[0]
    T = len(obs_seq)
    prev = np.zeros((T - 1, N), dtype=int)

    # DP matrix containing max likelihood of state at a given time
    V = np.zeros((N, T))
    V[:, 0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for n in range(N):
            seq_probs = V[:, t - 1] * A[:, n] * B[n, obs_seq[t]]
            prev[t - 1, n] = np.argmax(seq_probs)
            V[n, t] = np.max(seq_probs)

    return V, prev


def build_viterbi_path(prev, last_state):
    """Returns a state path ending in last_state in reverse order.
    最优路径回溯
    """
    T = len(prev)
    yield (last_state)
    for i in range(T - 1, -1, -1):
        yield (prev[i, last_state])
        last_state = prev[i, last_state]


def observation_prob(obs_seq):
    """ P( entire observation sequence | A, B, pi ) """
    return np.sum(forward(obs_seq)[:, -1])


def state_path(obs_seq, A, B, pi):
    """
    Returns
    -------
    V[last_state, -1] : float
        Probability of the optimal state path
    path : list(int)
        Optimal state path for the observation sequence
    """
    V, prev = viterbi(obs_seq, A, B, pi)
    # Build state path with greatest probability
    last_state = np.argmax(V[:, -1])
    path = list(build_viterbi_path(prev, last_state))

    return V[last_state, -1], reversed(path)

# 对应状态集合Q
states = ('Healthy', 'Fever')
# 对应观测集合V
observations = ('normal', 'cold', 'dizzy')
# 初始状态概率向量π
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
# 状态转移矩阵A
transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}
# 观测概率矩阵B
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}

# Index generation
states_id2label, states_label2id = generate_index_map(states)
observations_id2label, observations_label2id = generate_index_map(observations)

# Initial HMM structure
A = convert_map_to_matrix(transition_probability, states_label2id, states_label2id)
B = convert_map_to_matrix(emission_probability, states_label2id, observations_label2id)
observations_index = [observations_label2id[o] for o in observations]
pi = convert_map_to_vector(start_probability, states_label2id)

# generate simulation data
observations_data, states_data = simulate(20)
newA, newB, newpi = baum_welch_train(observations_data, A, B, pi)

states_out = state_path(observations_data, newA, newB, newpi)[1]
p = 0.0

print(states_data)
print(states_out)
for s in states_data:
    if next(states_out) == s:
        p += 1

print(p / len(states_data))



V, p = viterbi(observations_index, newA, newB, newpi)
print(" " * 7, " ".join(("%10s" % observations_id2label[i]) for i in observations_index))
for s in range(0, 2):
    print("%7s: " % states_id2label[s] + " ".join("%10s" % ("%f" % v) for v in V[s]))
print('\nThe most possible states and probability are:')
p, ss = state_path(observations_index, newA, newB, newpi)
for s in ss:
    print(states_id2label[s])
print(p)

