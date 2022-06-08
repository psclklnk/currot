import torch
import pickle
import gpytorch
import numpy as np


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
        self.training_iter = training_iter

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
