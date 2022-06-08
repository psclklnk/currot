import torch
import pickle
import gpytorch
import numpy as np
from scipy.spatial.distance import pdist
import nadaraya_watson as na


class Buffer:

    def __init__(self, n_elements, max_buffer_size, reset_on_query):
        self.reset_on_query = reset_on_query
        self.max_buffer_size = max_buffer_size
        self.buffers = [list() for i in range(0, n_elements)]

    def update_buffer(self, datas):
        if isinstance(datas[0], list):
            for buffer, data in zip(self.buffers, datas):
                buffer.extend(data)
        else:
            for buffer, data in zip(self.buffers, datas):
                buffer.append(data)

        while len(self.buffers[0]) > self.max_buffer_size:
            for buffer in self.buffers:
                del buffer[0]

    def read_buffer(self, reset=None):
        if reset is None:
            reset = self.reset_on_query

        res = tuple([buffer for buffer in self.buffers])

        if reset:
            for i in range(0, len(self.buffers)):
                self.buffers[i] = []

        return res

    def __len__(self):
        return len(self.buffers[0])


class Subsampler:

    def __init__(self, lb, ub, bins):
        eval_points = [np.linspace(lb[i], ub[i], bins[i] + 1)[:-1] for i in range(len(bins))]
        eval_points = [s + 0.5 * (s[1] - s[0]) for s in eval_points]
        self.bin_sizes = np.array([s[1] - s[0] for s in eval_points])
        self.eval_points = np.stack([m.reshape(-1, ) for m in np.meshgrid(*eval_points)], axis=-1)

    def __call__(self, discrete_sample):
        return self.eval_points[discrete_sample, :] + np.random.uniform(-0.5 * self.bin_sizes, 0.5 * self.bin_sizes)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RewardEstimatorGP:

    def __init__(self, training_iter=5):
        self.x = None
        self.y = None
        self.gp = None
        self.parameters = None
        self.training_iter = training_iter

    def set_parameters(self, parameters):
        self.parameters = parameters
        if self.gp is not None:
            self.gp.load_state_dict(parameters)

    def update_model(self, contexts, rewards):
        self.x = torch.from_numpy(contexts).float()  # .type(torch.float64)
        self.y = torch.from_numpy(rewards).float()  # .type(torch.float64)

        if self.gp is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp = ExactGPModel(self.x, self.y, likelihood)
        else:
            old_gp = self.gp
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp = ExactGPModel(self.x, self.y, likelihood)
            self.gp.load_state_dict(old_gp.state_dict())

        if self.parameters is None:
            self.gp.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.gp)
            for i in range(self.training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.gp(self.x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.y)
                loss.backward()
                optimizer.step()

            print('GP-Training Loss: %.3f' % (loss.item()))

            self.gp.eval()
            likelihood.eval()
        else:
            self.gp.load_state_dict(self.parameters)

    def predict_mean(self, samples):
        return torch.mean(self.gp(samples).mean)

    def predict_individual(self, samples, with_gradient=False):
        samples_torch = torch.from_numpy(samples).float()
        if with_gradient:
            samples_torch = samples_torch.requires_grad_(True)

        perf = self.gp(samples_torch).mean
        if with_gradient:
            grad = torch.autograd.grad(perf, samples_torch, grad_outputs=torch.ones_like(perf))[0]
            return perf.detach().numpy().astype(samples.dtype), grad.detach().numpy().astype(samples.dtype)
        else:
            return perf.detach().numpy().astype(samples.dtype)

    def __call__(self, samples, with_gradient=False):
        samples_torch = torch.from_numpy(samples).float()
        if with_gradient:
            samples_torch = samples_torch.requires_grad_(True)

        mean = self.predict_mean(samples_torch)
        if with_gradient:
            grad = torch.autograd.functional.jacobian(self.predict_mean, samples_torch)
            return mean.detach().numpy().astype(samples.dtype), grad.detach().numpy().astype(samples.dtype)
        else:
            return mean.detach().numpy().astype(samples.dtype)

    def save(self, path):
        data = self.training_iter, self.x, self.y
        if self.gp is not None:
            data = data + (self.gp.state_dict(),)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        if len(data) == 4:
            self.training_iter, self.x, self.y = data[:3]
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp = ExactGPModel(self.x, self.y, likelihood)
            self.gp.load_state_dict(data[3])
        else:
            self.training_iter, self.x, self.y = data


class NadarayaWatson:

    def __init__(self, contexts, returns, lengthscale=None, n_threads=5, n_max=None, radius_scale=3.):
        self.model = na.NadarayaWatson(contexts, returns, n_threads=n_threads)
        if lengthscale is None:
            self.lengthscale = np.median(pdist(contexts))
        else:
            self.lengthscale = lengthscale

        if n_max is None:
            self.n_max = int(0.5 * contexts.shape[0])
        else:
            self.n_max = n_max

        self.radius_scale = radius_scale

    def predict_individual(self, x):
        return np.reshape(self.model.predict(np.reshape(x, (-1, x.shape[-1])), self.lengthscale, n_max=self.n_max,
                                             radius_scale=self.radius_scale), x.shape[:-1])

    def save(self, path):
        pass

    def load(self, path):
        pass


class NadarayaWatsonPy:

    def __init__(self, contexts, returns, lengthscale=None):
        self.contexts = contexts
        self.returns = returns
        if lengthscale is None:
            self.lengthscale = 2 * (np.median(pdist(contexts)) ** 2)
        else:
            self.lengthscale = 2 * (lengthscale ** 2)

    def logsumexp(self, x):
        x_max = np.max(x, axis=-1, keepdims=True)
        return np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True)) + x_max

    def predict_individual(self, x, with_gradient=False):
        assert x.shape[-1] == self.contexts.shape[-1]

        # This allows for arbitrary batching dimensions at the beginning
        diffs = x[..., None, :] - self.contexts
        log_activations = -np.sum(np.square(diffs), axis=-1) / self.lengthscale
        weights = np.exp(log_activations - self.logsumexp(log_activations))
        weighted_returns = weights * self.returns
        prediction = np.sum(weights * self.returns, axis=-1)

        if with_gradient:
            la_grads = -(2 / self.lengthscale) * diffs
            avg_la_grad = np.einsum("...i,...ij->...j", weights, la_grads)

            return prediction, np.einsum("...i,...ij->...j", weighted_returns, la_grads) - \
                   np.einsum("ij,i->ij", avg_la_grad, prediction)
        else:
            return prediction

    def save(self, path):
        pass

    def load(self, path):
        pass
