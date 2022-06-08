import os
import torch
import pickle
import numpy as np
import cyipopt as ipopt
from geomloss import SamplesLoss


class WassersteinInterpolation:

    def __init__(self, init_samples, target_sampler, perf_lb, epsilon, callback=None, opt_tol=1e-3, opt_time=2.,
                 ws_scaling=0.9, ws_blur=0.01):
        self.current_samples = init_samples
        self.n_samples, self.dim = self.current_samples.shape
        self.target_sampler = target_sampler

        self.perf_lb = perf_lb
        self.epsilon = epsilon

        self.sl = SamplesLoss("sinkhorn", blur=ws_blur, scaling=ws_scaling, backend="tensorized")
        self.callback = callback

        self.target_plan = None
        self.model = None
        self.opt_mask = None

        lb = [0.] * self.n_samples
        ub = [1.] * self.n_samples

        cl = [-np.inf] + [perf_lb] * self.n_samples
        cu = [epsilon] + [np.inf] * self.n_samples
        self.nlp = ipopt.Problem(n=self.n_samples, m=len(cl), problem_obj=self, lb=lb, ub=ub, cl=cl, cu=cu)

        # ... and configure some options of it (e.g. that we do not need to be so perfectly accurate,
        # after all it is RL)
        self.nlp.add_option('mu_strategy', 'adaptive')
        self.nlp.add_option('hessian_approximation', 'limited-memory')
        self.nlp.add_option('print_level', 0)
        self.nlp.add_option('max_cpu_time', opt_time)
        self.nlp.add_option('tol', opt_tol)

    def wasserstein_distance(self, initial_samples, target_samples, for_backprop=False):
        x = torch.from_numpy(initial_samples).requires_grad_(True)
        alpha = torch.ones(x.shape[0], dtype=x.dtype) / x.shape[0]
        y = torch.from_numpy(target_samples)
        beta = torch.ones(y.shape[0], dtype=y.dtype) / y.shape[0]

        if for_backprop:
            return self.sl(alpha, x, beta, y), x
        else:
            return self.sl(alpha, x, beta, y).detach().numpy()

    def compute_transport_plan(self, initial_samples, target_samples):
        wdist, x = self.wasserstein_distance(initial_samples, target_samples, for_backprop=True)

        g, = torch.autograd.grad(wdist, [x])
        return -(g * x.shape[0]).detach().numpy()

    @staticmethod
    def _distance(target_plan, x, grad=False):
        x = np.clip(x, 0., 1.)
        dist = x[:, None] * target_plan
        squared_dists = np.sum(np.square(dist), axis=-1)

        if grad:
            return np.einsum("ij,ij->i", dist, target_plan) / squared_dists.shape[0]
        else:
            return 0.5 * np.mean(squared_dists)

    @staticmethod
    def _particle_performance(initial_samples, target_plan, model, x, grad=False):
        x = np.clip(x, 0., 1.)
        moved_samples = initial_samples + x[:, None] * target_plan
        if grad:
            # Preds [N], Grads [N, D]
            pers, perf_grads = model.predict_individual(moved_samples, with_gradient=True)
            return np.einsum("ij,ij->i", perf_grads, target_plan)
        else:
            return model.predict_individual(moved_samples, with_gradient=False)

    def objective(self, x):
        x = np.clip(x, 0., 1.)
        dist = (1 - x[:, None]) * self.target_plan
        squared_dists = np.where(self.opt_mask, np.sum(np.square(dist), axis=-1), 0.)
        return 0.5 * np.mean(squared_dists)

    def gradient(self, x):
        x = np.clip(x, 0., 1.)
        dist = (1 - x[:, None]) * self.target_plan
        return np.where(self.opt_mask, -np.einsum("ij,ij->i", dist, self.target_plan) / dist.shape[0], 0.)

    def constraints(self, x):
        c1 = self._distance(self.target_plan, x, grad=False)
        c2 = self._particle_performance(self.current_samples, self.target_plan, self.model, x, grad=False)
        c2 = np.where(self.opt_mask, c2, self.perf_lb)
        return np.concatenate([[c1], c2])

    def jacobian(self, x):
        j1 = self._distance(self.target_plan, x, grad=True)
        j2 = self._particle_performance(self.current_samples, self.target_plan, self.model, x,
                                        grad=True)
        j2 = np.where(self.opt_mask, j2, 0.)
        return np.concatenate((j1[None, :], np.diag(j2)), axis=0)

    def update_distribution(self, model, success_samples):
        # We reset the current samples to the success samples, if they are below the performance threshold
        init_samples = self.current_samples.copy()
        low_perfs = model.predict_individual(init_samples) < self.perf_lb
        if np.any(low_perfs):
            value_plan = self.compute_transport_plan(init_samples, success_samples)
            init_samples[low_perfs, :] = init_samples[low_perfs, :] + value_plan[low_perfs, :]
        target_samples = self.target_sampler(self.n_samples)

        # Now we optimize the distance to the target while trying to fulfill the
        opt_mask = model.predict_individual(init_samples) > self.perf_lb
        if np.sum(opt_mask) == 0:
            print("Skipping optimization as no particle is above the performance constraint")
            new_samples = init_samples
        else:
            self.opt_mask = opt_mask
            self.model = model
            self.target_plan = self.compute_transport_plan(init_samples, target_samples)

            # Do a fine-tuning by optimizing the coefficients with IPOPT (should take close to no time)
            import time
            t1 = time.time()
            x, info = self.nlp.solve(np.zeros(self.n_samples))
            t2 = time.time()
            x = np.clip(x, 0., 1.)
            new_samples = init_samples + x[:, None] * self.target_plan
            print("Optimization took: %.3e s" % (t2 - t1))

        print("Optimized performance: %.3e" % model(new_samples))
        print("Optimized target distance: %.3e" % self.wasserstein_distance(new_samples, target_samples))

        if self.callback is not None:
            self.callback(init_samples, new_samples, success_samples, target_samples)

        self.current_samples = new_samples

    def save(self, path):
        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump((self.current_samples, self.perf_lb, self.epsilon), f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            tmp = pickle.load(f)

            self.current_samples = tmp[0]
            self.n_samples = self.current_samples.shape[0]

            self.perf_lb = tmp[1]
            self.epsilon = tmp[2]
