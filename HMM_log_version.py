import numpy as np
from scipy.misc import logsumexp
from util import print_progress



class hmm_log(object):
    def __init__(self, n, m):
        # initialize model
        self.N = n
        self.M = m

        self.A = np.triu(np.random.uniform(1e-10, 1, (self.N, self.N)))
        self.A = self.A / np.sum(self.A, axis=1)[:, np.newaxis]
        self.B = np.random.uniform(1e-10, 1, (self.N, self.M))
        self.B = self.B / np.sum(self.B, axis=1)[:, np.newaxis]
        self.pi = np.zeros([self.N], dtype=np.float64)
        self.pi[0] = 1.0

        # if using log likelihood:
        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)
        self.log_pi = np.log(self.pi)

        self.niter = 100
        self.eps = 1e-4

    def set_param(self, A, B, p):
        self.A = A
        self.B = B
        self.pi = p
        self.log_A = np.log(A)
        self.log_B = np.log(B)
        self.log_pi = np.log(p)

    def inference_forward_log(self, obs):
        """
        Return the log-likelihood version of the forward procedure
        :param obs: [T,] observation sequence, each element lies in the range [0, M)
        :return: log_prob, log_alpha
        """
        T = np.shape(obs)[0]
        # initialization
        log_alpha = np.zeros([self.N, T])
        log_alpha[:, 0] = self.log_pi+self.log_B[:, obs[0]]
        # induction
        for t in range(1, T):
            log_alpha[:, t] = logsumexp(log_alpha[:, t-1][:, np.newaxis] + self.log_A, axis=0) + self.log_B[:, obs[t]]

        # termination
        return log_alpha, logsumexp(log_alpha[:, -1])

    def inference_backward_log(self, obs):
        """
        Return the log-likelihood version of the backward procedure
        :param obs: [T,] observation sequence, each element lies in the range [0, M)
        :return: log_prob, log_beta
        """
        T = np.shape(obs)[0]
        # initialization
        log_beta = np.zeros([self.N, T])
        # induction
        for t in range(T-2, -1, -1):
            log_beta[:, t] = logsumexp(self.log_A + (self.log_B[:, obs[t+1]] + log_beta[:, t+1]), axis=1)

        return log_beta

    def baum_welch_log(self, obs):
        """
        Baum_welch algorithm - log version
        :param obs:
        :return: [T,] observation sequence, each element lies in the range [0, M)
        """
        T = np.shape(obs)[0]
        log_likelihood = 0
        # initialization
        prev_log_likelihood = 0
        log_xi = np.zeros([self.N, self.N, T-1])

        for i in range(self.niter):
            # predict
            log_alpha, log_likelihood = self.inference_forward_log(obs)
            log_beta = self.inference_backward_log(obs)
            log_gamma = log_alpha + log_beta
            log_gamma = log_gamma - logsumexp(log_gamma, axis=0)
            for t in range(T-1):
                temp = self.log_A + log_alpha[:, t][:, np.newaxis] + \
                       (self.log_B[:, obs[t+1]] + log_beta[:, t+1])[np.newaxis, :]
                log_xi[:, :, t] = temp - logsumexp(temp)

            # terminate condition
            if i >=10 and abs(prev_log_likelihood - log_likelihood) < self.eps * abs(log_likelihood):
                break

            prev_log_likelihood = log_likelihood

            # update parameters
            #self.log_pi = log_gamma[:, 0]   # do not update pi for left to right model
            self.log_A = logsumexp(log_xi, axis=-1) - logsumexp(log_gamma[:, 0:T], axis=-1)[:, np.newaxis]
            gsum = logsumexp(log_gamma, axis=-1)
            for m in range(self.M):
                if (obs == m).any():
                    self.log_B[:, m] = logsumexp(log_gamma[:, obs==m], axis=1) - gsum
                else:
                    self.log_B[:, m] = - gsum

            print('iter {} {}'.format(i, log_likelihood))

        self.A = np.exp(self.log_A)
        self.B = np.exp(self.log_B)
        return log_likelihood

    def baum_welch_log_multi(self, obs_set):
        """
        Baum-Welch method for multiple observation sequences - log version
        :param obs_set: list of np vectors
        :return: array of log probability for each sequence
        """
        num_seq = len(obs_set)
        prev_log_probs = np.zeros([num_seq])
        log_probs = np.zeros([num_seq])
        log_A_upper = np.zeros_like(self.A)
        log_A_lower = np.zeros([self.N])
        log_B_upper = np.zeros_like(self.B)
        log_B_lower = np.zeros([self.N])

        for i in range(self.niter):
            # print_progress(i, self.niter, prefix='Num of iterations', suffix='')
            print("iteration {}".format(i))
            # clear cache
            log_A_upper.fill(0)
            log_A_lower.fill(0)
            log_B_upper.fill(0)
            log_B_lower.fill(0)
            for k, obs in enumerate(obs_set):
                T = np.shape(obs)[0]
                log_xi = np.zeros([self.N, self.N, T - 1])
                # predict
                log_prob, log_alpha = self.inference_forward_log(obs)
                log_beta = self.inference_backward_log(obs)[1]
                log_gamma = log_alpha + log_beta
                log_gamma = log_gamma - logsumexp(log_gamma, axis=0)
                for t in range(T - 1):
                    C = self.log_A + log_alpha[:, t][:, np.newaxis] + \
                        self.log_B[:, obs[t + 1]][np.newaxis, :] + log_beta[:, t + 1][np.newaxis, :]
                    log_xi[:, :, t] = C - logsumexp(C)

                # update parameters
                log_A_upper_k = logsumexp(log_xi, axis=-1)
                log_A_lower_k = logsumexp(log_gamma[:, 0:T], axis=-1)
                log_B_upper_k = np.zeros_like(self.log_B) - np.inf
                for m in range(self.M):
                    if np.shape(log_gamma[:, obs == m])[1] > 0:
                        log_B_upper_k[:, m] = logsumexp(log_gamma[:, obs == m], axis=1)
                log_B_lower_k = logsumexp(log_gamma, axis=-1)

                log_A_upper = logsumexp(np.stack((log_A_upper, log_A_upper_k), axis=-1), axis=-1)
                log_A_lower = logsumexp(np.stack((log_A_lower, log_A_lower_k), axis=-1), axis=-1)
                log_B_upper = logsumexp(np.stack((log_B_upper, log_B_upper_k), axis=-1), axis=-1)
                log_B_lower = logsumexp(np.stack((log_B_lower, log_B_lower_k), axis=-1), axis=-1)

                log_probs[k] = log_prob

            # terminate condition
            if np.abs(np.sum(log_probs)-np.sum(prev_log_probs)) < self.eps * np.sum(log_probs):
                break

            prev_log_probs = log_probs
            print(log_probs)
            self.log_A = log_A_upper - log_A_lower[:, np.newaxis]
            self.log_B = log_B_upper - log_B_lower[:, np.newaxis]

        return log_probs