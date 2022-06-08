import numpy as np


class Exp3S:

    def __init__(self, n, eta, eps=0.2, norm_hist_len=1000):
        self.weights = np.zeros(n)
        self.eta = eta
        self.norm_hist_len = norm_hist_len
        self.max_reward = -np.inf
        self.min_reward = np.inf
        self.eps = eps
        self.count = 1

    # We use a normalization via max and min as keeping a reservoir sometimes led to instabilities of the algorithm,
    # if the learner faced longer periods without progress. Keeping max- and min-values avoids this collapse as we can
    # take the whole history for computing the into account. Note that this only changes the scale of the rewards.
    def normalize_reward(self, r):
        if r > self.max_reward:
            self.max_reward = r

        if r < self.min_reward:
            self.min_reward = r

        # Avoid division by zero and hence numerical problems in the update of the context distribution
        if np.isclose(self.min_reward, self.max_reward):
            return 0.
        else:
            r_clip = np.clip(r, self.min_reward, self.max_reward)
            return 2 * ((r_clip - self.min_reward) / (self.max_reward - self.min_reward)) - 1

    @staticmethod
    def logsumexp(w):
        w_max = np.max(w)
        return np.log(np.sum(np.exp(w - w_max))) + w_max

    def update(self, i, r):
        # Normalize the reward
        r_norm = self.normalize_reward(r)

        # Update the weights:
        #   1. Add the reward for the chosen arm (in the paper by Graves et al, a \beta parameter is described. In
        #      their experiments, they however set it to zero and the original Exp3.S algorithm from Auer et al. does
        #      not have such a parameter. Hence we leave it out in this implementation
        cur_probs = np.exp(self.weights - self.logsumexp(self.weights))
        self.weights[i] += self.eta * (r_norm / cur_probs[i])

        #   2. Initial diversification that decays over time. Note that we again rather use the Exp3.S algorithm
        #      described by Auer et al., which has a minor difference in this diversification. However, given that
        #      curricula process thousands of task, this difference is negligible in practice as these extra
        #      terms quickly decay. The update of the likelihoods is done in log-space for numerical stability
        self.count += 1
        alpha_t = 1. / self.count
        log_t1 = self.weights + np.log(1 - alpha_t)
        log_t2 = self.logsumexp(self.weights) + np.log(alpha_t) - np.log(self.weights.shape[0])
        log_t_max = np.maximum(log_t1, log_t2)
        self.weights = np.log(np.exp(log_t1 - log_t_max) + np.exp(log_t2 - log_t_max)) + log_t_max

    def sample(self):
        # Sample according to an epsilon greedy policy
        policy_probs = np.exp(self.weights - self.logsumexp(self.weights))
        probs = (1 - self.eps) * policy_probs + self.eps / self.weights.shape[0]
        return np.random.choice(np.arange(self.weights.shape[0]), p=probs)
