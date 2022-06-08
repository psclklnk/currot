import numpy as np
from deep_sprl.teachers.abstract_teacher import AbstractTeacher


class DistributionSampler(AbstractTeacher):

    def __init__(self, sample_func, lb, ub):
        self.sample_func = sample_func
        self.lb = lb
        self.ub = ub
        self.dim = self.sample_func(n=1).shape[1]

        n_samples = int(1e6)
        samples_not_ok = np.ones(n_samples, dtype=np.bool)
        samples = np.zeros((n_samples, self.dim))
        while np.any(samples_not_ok):
            n_new_samples = np.sum(samples_not_ok)
            samples[samples_not_ok] = self.sample_func(n=n_new_samples)
            samples_not_ok = np.logical_or(np.any(samples < lb, axis=-1),
                                           np.any(samples > ub, axis=-1))

        self.mu = np.mean(samples, axis=0)
        diffs = (samples - self.mu[None, :]) / np.sqrt(samples.shape[0])
        self.cov = np.einsum("ni,nj->ij", diffs, diffs)

    def sample(self):
        ok = False
        while not ok:
            sample = self.sample_func(n=1)[0, :]
            ok = np.all(sample >= self.lb) and np.all(sample <= self.ub)
        return sample

    def mean(self):
        return self.mu

    def covariance_matrix(self):
        return self.cov

    def save(self, path):
        pass

    def load(self, path):
        pass


class DiscreteSampler(AbstractTeacher):

    def __init__(self, log_likelihoods):
        self.likelihoods = np.exp(log_likelihoods)

    def sample(self):
        return np.argmax(np.random.uniform(0., 1.) <= np.cumsum(self.likelihoods))

    def save(self, path):
        pass

    def load(self, path):
        pass


class UniformSampler(AbstractTeacher):

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self):
        norm_sample = np.random.uniform(low=-1, high=1, size=self.lower_bound.shape)
        return self._scale_context(norm_sample)

    def mean(self):
        return 0.5 * self.lower_bound + 0.5 * self.upper_bound

    def covariance_matrix(self):
        return np.diag((0.5 * (self.upper_bound - self.lower_bound)) ** 2)

    def _scale_context(self, context):
        b = 0.5 * (self.upper_bound + self.lower_bound)
        m = 0.5 * (self.upper_bound - self.lower_bound)
        return m * context + b

    def save(self, path):
        pass

    def load(self, path):
        pass
