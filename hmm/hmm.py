import numpy as np
import pandas as pd



class HMM:
    """
    Attributes:
    ------------
    A : pandas.DataFrame
        State transition probability matrix with shape (number of hidden states, number of hidden states)

    B : pandas.DataFrame
        Emission probability matrix with shape (number of hidden states, number of observation states)

    pi: list
        Initial hidden states distributions

    """

    def __init__(self, A, B, pi, obs, hid):
        self.A = A
        self.B = B
        self.pi = pi
        self.Q_states = list()
        self.Q_probs = list()
        self.Y_states = list()
        self.Y_probs = list()
        self.nb_obs_states = np.arange(len(obs))
        self.nb_hid_states = np.arange(len(hid))

    def sample(self, T):
        A = self.A.get_values()
        B = self.B.get_values()

        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        hiddens = np.zeros(T, dtype=int)

        hiddens[0] = draw_from(self.pi)
        observations[0] = draw_from(B[hiddens[0], :])

        for t in range(1, T):
            hiddens[t] = draw_from(A[hiddens[t-1], :])
            observations[t] = draw_from(B[hiddens[t-1], :])
        return hiddens, observations

        #
        # for t in range(T):
        #     q_t = np.dot(self.Q[-1], A)
        #     print(q_t)
        #     self.Q.append(q_t.tolist())
        #     y_t = np.dot(q_t, B)
        #     print(y_t)
        #
        # print(self.Q)

    def forward(self, observations):
        T = len(observations)
        A = self.A.get_values()
        B = self.B.get_values()
        n_hid, n_obs = B.shape

        alphas = np.zeros((n_hid, T))
        alphas[:, 0] = self.pi * B[:, observations[0]]
        for t in range(1, T):
            for i in range(n_hid):
                alphas[i, t] = np.dot(alphas[:, t-1], A[:, i]) * B[i, observations[t]]

        return alphas, sum(alphas[:, -1])

    def backward(self, observations):
        T = len(observations)
        A = self.A.get_values()
        B = self.B.get_values()
        n_hid, n_obs = B.shape

        betas = np.zeros((n_hid, T))
        betas[:, -1] = 1

        for t in reversed(range(T-1)):
            for i in range(n_hid):
                betas[i, t] = np.sum(betas[:,t+1] * A[i,:] * B[:, observations[t+1]])

        # prob = 0.0
        # for i in range(n_hid):
        #     prob += self.pi[i] * B[i, observations[0]] * betas[i, 0]
        # print(prob)
        return betas, np.sum(self.pi * B[:, observations[0]]* betas[:, 0])



def build_A_B_matrixs(obs_state_list, hid_state_list,
                      trainsition_probability_map, emission_probability_map,
                      disp=True):

    def build_bi_directory(labels):
        to_lbl, to_idx = dict(), dict()
        for i, lbl in enumerate(labels):
            to_lbl[i] = lbl
            to_idx[lbl] = i
        return to_lbl, to_idx

    obs_idx_2_lbl, obs_lbl_2_idx = build_bi_directory(obs)
    hid_idx_2_lbl, hid_lbl_2_idx = build_bi_directory(hid)
    A = pd.DataFrame.from_dict(transition_probability, orient='index')
    B = pd.DataFrame.from_dict(emission_probability, orient='index')
    if disp:
        print(A)
        print(B)

    def convert_col_name(df, label_idx_map):
        col_names = []
        for c in df.columns:
            id = label_idx_map[c]
            col_names.append(id)
        df.columns = col_names

    def convert_idx_name(df, label_idx_map):
        idx_names = []
        for i in df.index:
            id = label_idx_map[i]
            idx_names.append(id)
        df.index = idx_names
        df.sort_index(inplace=True)

    convert_col_name(A, hid_lbl_2_idx)
    convert_idx_name(A, hid_lbl_2_idx)
    convert_col_name(B, obs_lbl_2_idx)
    convert_idx_name(B, hid_lbl_2_idx)

    if disp:
        print("observations state -> id")
        for k, v in obs_lbl_2_idx.items():
            print(k, v)
        print("hidden state -> id")
        for k, v in hid_lbl_2_idx.items():
            print(k, v)

    return A, B, dict(obs_idx_2_lbl=obs_idx_2_lbl,
                      obs_lbl_2_idx=obs_lbl_2_idx,
                      hid_idx_2_lbl=hid_idx_2_lbl,
                      hid_lbl_2_idx=hid_lbl_2_idx)

def build_pi(start_probability, hid_idx_to_label):
    ret = []
    for _, name in hid_idx_to_label.items():
        value = start_probability.get(name, 0.0)
        ret.append(value)
    return ret



if __name__ == '__main__':


    # example refer to: https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    obs = ('normal', 'cold', 'dizzy')
    hid = ('healthy', 'fever')

    start_probability = dict(healthy=0.6, fever=0.4)
    # start_probability = dict(fever=1.0)

    transition_probability = dict(
        healthy = dict(healthy=0.7, fever=0.3),
        fever = dict(healthy=0.4, fever=0.6)
    )
    emission_probability = dict(
        healthy = dict(normal=0.5, cold=0.4, dizzy=0.1),
        fever = dict(normal=0.1, cold=0.3, dizzy=0.6)
    )

    A, B, lookup = build_A_B_matrixs(obs, hid, transition_probability, emission_probability, False)
    pi = build_pi(start_probability, lookup['hid_idx_2_lbl'])
    # print(A)
    # print(B)
    # print("Observations")
    # print(lookup['obs_idx_2_lbl'])
    # print(lookup['obs_lbl_2_idx'])
    # print("Hidden")
    # print(lookup['hid_idx_2_lbl'])
    # print(lookup['hid_lbl_2_idx'])
    # print(pi)

    hmm = HMM(A, B, pi, obs, hid)
    q, y = hmm.sample(4)
    print(q)
    print(y)

    alphas, prob1 = hmm.forward(y)
    betas, prob2 = hmm.backward(y)
    print(prob1)
    print(prob2)



