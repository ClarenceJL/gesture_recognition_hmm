import numpy as np
from util import print_progress


class hmm(object):
    def __init__(self, n, m, use_scaling=True):
        # initialize model
        self.N = n
        self.M = m

        self.A = np.triu(np.random.uniform(1e-10, 1, (self.N, self.N)))
        self.A = self.A / np.sum(self.A, axis=1)[:, np.newaxis]
        self.B = np.random.uniform(1e-10, 1, (self.N, self.M))
        self.B = self.B / np.sum(self.B, axis=1)[:, np.newaxis]
        self.pi = np.zeros([self.N], dtype=np.float64)
        self.pi[0] = 1.0

        self.use_scaling = use_scaling
        self.niter = 100
        self.niter_multi = 12
        self.eps = 1e-5

    def set_param(self, A, B, p):
        self.A = A
        self.B = B
        self.pi = p

    def inference_forward(self, obs):
        T = np.shape(obs)[0]
        # initialization
        alpha = np.zeros([self.N, T])
        c_inv = np.zeros([T])
        alpha[:, 0] = self.pi*self.B[:, obs[0]]
        c_inv[0] = np.sum(alpha[:, 0])
        if self.use_scaling and c_inv[0] > 0:
            alpha[:, 0] = alpha[:, 0] / c_inv[0]
        # induction
        for t in range(1, T):
            alpha[:, t] = np.dot(np.reshape(alpha[:, t-1], (1, -1)), self.A) * self.B[:, obs[t]]
            c_inv[t] = np.sum(alpha[:, t])
            if self.use_scaling and c_inv[t] > 0:
                alpha[:, t] = alpha[:, t] / c_inv[t]

        log_likelihood = np.sum(np.log(c_inv[c_inv>=0]))

        c_inv = np.where(c_inv > 0, c_inv, 1e-100)
        return alpha, c_inv, log_likelihood

    def inference_backward(self, obs, c_inv=[]):
        T = np.shape(obs)[0]
        # initialization
        beta = np.ones([self.N, T])
        if self.use_scaling:
            if len(c_inv) > 0:
                beta[:, -1] = beta[:, -1] / c_inv[-1]
            elif np.sum(beta[:, -1]) > 0:
                beta[:, -1] = beta[:, -1] / np.sum(beta[:, -1])
        # induction
        for t in range(T-2, -1, -1):
            beta[:, t] = np.dot(self.A, np.reshape(self.B[:, obs[t+1]] * beta[:, t+1], [-1, 1])).flatten()
            if self.use_scaling:
                if len(c_inv) > 0:
                    beta[:, t] = beta[:, t] / c_inv[t]
                elif np.sum(beta[:, t]) > 0:
                    beta[:, t] = beta[:, t] / np.sum(beta[:, t])

        return beta

    def state_probability(self, alpha, beta):
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=0)
        return gamma

    def baum_welch(self, obs):
        T = np.shape(obs)[0]
        # initialization
        prev_log_prob = 0.0
        xi = np.zeros([self.N, self.N, T-1])

        for i in range(self.niter):
            #print_progress(i, self.niter, prefix='', suffix='')
            # predict
            alpha, c_inv, log_prob = self.inference_forward(obs)
            beta = self.inference_backward(obs, c_inv)
            gamma = self.state_probability(alpha, beta)

            for t in range(T-1):
                xi[:, :, t] = self.A * alpha[:, t][:, np.newaxis] * \
                              (self.B[:, obs[t+1]] * beta[:, t+1])[np.newaxis, :]
                xi[:, :, t] = xi[:, :, t] / np.sum(xi[:, :, t])

            # terminate condition
            if abs(log_prob - prev_log_prob) < self.eps * abs(prev_log_prob):
                 break

            prev_log_prob = log_prob

            # update parameters
            # self.pi = gamma[:, 0]  # do not update pi for left to right model
            self.A = np.sum(xi, axis=2) / np.sum(gamma[:, 0:T], axis=1)[:, np.newaxis]
            gsum = np.sum(gamma, axis=1)
            for k in range(self.M):
                self.B[:, k] = np.sum(gamma[:, obs==k], axis=1) / gsum

            self.B[self.B < 1e-100] = 1e-100
            print('iter {} {}'.format(i, log_prob))

        return log_prob


    def baum_welch_multi(self, obs_set):
        """
        Baum-Welch method for multiple observation sequences
        :param obs_set: list of np vectors
        :return: array of log probability for each sequence
        """
        num_seq = len(obs_set)
        prev_log_probs = np.ones([num_seq])
        log_probs = np.zeros([num_seq])
        A_upper = np.zeros_like(self.A)
        A_lower = np.zeros([self.N])
        B_upper = np.zeros_like(self.B)
        B_lower = np.zeros([self.N])

        for i in range(self.niter_multi):
            A_upper.fill(0)
            A_lower.fill(0)
            B_upper.fill(0)
            B_lower.fill(0)

            for k, obs in enumerate(obs_set):
                T = np.shape(obs)[0]
                xi = np.zeros([self.N, self.N, T - 1])
                # predict
                alpha, c, log_probs[k] = self.inference_forward(obs)
                alpha[alpha < 1e-150] = 1e-150
                beta = self.inference_backward(obs)
                beta[beta < 1e-150] = 1e-150
                gamma = self.state_probability(alpha, beta)
                for t in range(T - 1):
                    xi[:, :, t] = self.A * alpha[:, t][:, np.newaxis] * \
                                  (self.B[:, obs[t + 1]] * beta[:, t + 1])[np.newaxis, :]
                    xi[:, :, t] = xi[:, :, t] / np.sum(xi[:, :, t])

                # update parameters
                A_upper = A_upper + np.sum(xi, axis=2)
                A_lower = A_lower + np.sum(gamma[:, 0:T], axis=1)
                for m in range(self.M):
                    B_upper[:, m] = B_upper[:, m] + np.sum(gamma[:, obs == m], axis=1)
                B_lower = B_lower + np.sum(gamma, axis=1)


            # terminate condition
            if np.abs(np.sum(log_probs)-np.sum(prev_log_probs)) < self.eps * np.sum(log_probs):
                break

            prev_log_probs = log_probs
            self.A = A_upper / A_lower[:, np.newaxis]
            self.B = B_upper / B_lower[:, np.newaxis]

            self.B[self.B < 1e-100] = 1e-100

            print('iter {} {}'.format(i, np.mean(log_probs)))

        return log_probs