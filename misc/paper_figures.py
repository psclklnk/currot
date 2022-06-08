import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize, NonlinearConstraint
from deep_sprl.teachers.spl.wasserstein_interpolation import WassersteinInterpolation

FONT_SIZE = 8
TICK_SIZE = 6

plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\usepackage{amssymb}'
                              r'\usepackage{amsthm}'
                              r'\usepackage{amsfonts}'
                              r'\DeclareMathOperator*{\argmin}{arg\,min}')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})


def gauss_log_pdf(xs, mean, var):
    logs = -0.5 * (np.square(xs - mean) / var)
    return logs - logsumexp(logs)


def compute_projection(xs, mu, values, delta):
    def objective(params):
        log_pdf = gauss_log_pdf(xs, params[0], params[1])
        pdf = np.exp(log_pdf)
        return np.sum(pdf * (log_pdf - np.log(mu)))

    def perf_con(params):
        log_pdf = gauss_log_pdf(xs, params[0], params[1])
        return np.sum(np.exp(log_pdf) * values)

    con = NonlinearConstraint(perf_con, delta, np.inf)
    res = minimize(objective, np.array([0.5, 0.25]), method="trust-constr", constraints=[con],
                   bounds=[(0., 1.), (1e-10, 50.)])
    return res.x


class ValueFunction(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            return np.mean(self.net(torch.from_numpy(x).float()).detach().numpy().astype(x.dtype))
        else:
            return self.net(x)

    def predict_individual(self, samples, with_gradient=False):
        samples_torch = torch.from_numpy(samples).float()
        if with_gradient:
            samples_torch = samples_torch.requires_grad_(True)

        perf = self.net(samples_torch)
        if with_gradient:
            grad = torch.autograd.grad(perf, samples_torch, grad_outputs=torch.ones_like(perf))[0]
            return np.squeeze(perf.detach().numpy().astype(samples.dtype)), \
                   grad.detach().numpy().astype(samples.dtype)
        else:
            return np.squeeze(perf.detach().numpy().astype(samples.dtype))


def train_vf(vf, data_x, data_y, max_iter=10000):
    data_x = torch.from_numpy(data_x).float()[:, None]
    data_y = torch.from_numpy(data_y).float()

    optimizer = torch.optim.Adam(vf.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    rel_err = torch.mean(torch.abs(data_y - torch.squeeze(vf.forward(data_x))) / torch.clamp_min(torch.abs(data_y), 1.))
    count = 0
    while rel_err > 0.05 and count < max_iter:
        loss = loss_fn(torch.squeeze(vf.forward(data_x)), data_y)
        print("Relative Error: %.3e" % rel_err)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rel_err = torch.mean(
            torch.abs(data_y - torch.squeeze(vf.forward(data_x))) / torch.clamp_min(torch.abs(data_y), 1.))
        count += 1


def get_vf():
    torch.random.manual_seed(0)
    vf = ValueFunction()
    np.random.seed(1)
    train_x = np.random.uniform(0, 1, 10)
    train_y = np.minimum(1., 1 / (10 * train_x))
    train_vf(vf, train_x, train_y)
    return vf


def logsumexp(x, axis=None):
    x_max = np.max(x, axis=axis)
    return np.log(np.sum(np.exp(x - x_max), axis=axis)) + x_max


def expected_value(target_ll, values, eta):
    interp_ll = target_ll + eta * values
    interp_ll -= logsumexp(interp_ll)
    return np.sum(np.exp(interp_ll) * values)


def kl_interpolation_plot1(path=None):
    xs = np.linspace(0, 1, 1000)
    target_lls = []
    deltas = [0.95, 0.85, 0.3, 0.1, 0.]

    target_ll = np.log(np.where(xs > 0.75, 1, 1e-5))
    target_ll -= logsumexp(target_ll)
    target_lls.append(target_ll)

    target_ll = -10 * np.square(xs - 0.9)
    target_ll -= logsumexp(target_ll)
    target_lls.append(target_ll)

    # Value function (just a randomly trained NN)
    vf = get_vf()
    values = np.squeeze(vf(torch.from_numpy(xs).float()[:, None]).detach().numpy())
    values /= np.max(values)

    # Generate the different plots for the different eta
    f = plt.figure(figsize=(6, 1.5))
    for row, target_ll in enumerate(target_lls):
        axs = [f.add_axes([i / len(deltas) + 0.015, 0.08 + 0.35 * row, 1 / len(deltas) - 0.03, 0.33]) for i in
               range(len(deltas))]
        for i, delta in enumerate(deltas):
            # Compute the right value of eta
            if expected_value(target_ll, values, 0) >= delta:
                eta = 0.
            else:
                eta = brentq(lambda x: expected_value(target_ll, values, x) - delta, 2000., 0.)
            interp_ll = target_ll + eta * values
            interp_ll -= logsumexp(interp_ll)

            axs[i].hlines(delta, 0, 1, colors="black", linestyle="--")

            vl, = axs[i].plot(xs, values, color="C1")
            axs[i].fill_between(xs, 0, values, color="C1", alpha=0.3)

            tl, = axs[i].plot(xs, np.exp(target_ll) / np.exp(np.max(target_ll)), color="C0")
            axs[i].fill_between(xs, 0., np.exp(target_ll) / np.exp(np.max(target_ll)), color="C0", alpha=0.5)
            il, = axs[i].plot(xs, np.exp(interp_ll) / np.exp(np.max(interp_ll)), color="C2")
            axs[i].fill_between(xs, 0., np.exp(interp_ll) / np.exp(np.max(interp_ll)), color="C2", alpha=0.5)

            gparams = compute_projection(xs, np.exp(target_ll), values, delta)
            gls = np.exp(gauss_log_pdf(xs, gparams[0], gparams[1]))
            gl, = axs[i].plot(xs, gls / np.max(gls), color="C5")
            axs[i].fill_between(xs, 0., gls / np.max(gls), color="C5", alpha=0.5)

            # axs[i].legend([vl, tl, il], [r"$J(\pi, c)$", r"$\mu(c)$", r"$\mu(c)\exp(J(\pi, c))^{\eta}$"],
            #               fontsize=TICK_SIZE)
            if row == 1:
                axs[i].set_title(r"$\delta=%.2f$" % delta, fontsize=FONT_SIZE, pad=0.)

            if row == 0:
                axs[i].set_xlabel(r"$c$", fontsize=FONT_SIZE, labelpad=-1)

            axs[i].set_xticklabels([])
            axs[i].grid()
            axs[i].set_xlim([0, 1])
            axs[i].set_xticks([0., 0.25, 0.5, 0.75, 1.])
            axs[i].set_ylim([0, 1])
            axs[i].set_yticks([0., 0.25, 0.5, 0.75, 1.])
            axs[i].set_yticklabels([])
            axs[i].tick_params('both', length=0, width=0, which='major')

            if i == 0:
                # axs[i].set_yticks([0., 1.])
                # axs[i].set_yticklabels(["0", "1"], fontsize=TICK_SIZE)
                axs[i].tick_params('x', length=0, width=0, which='major')
                axs[i].tick_params('y', which="major", pad=0)

                if row == 1:
                    axs[i].legend([vl, tl, il, gl], [r"$J(\pi, c)$", r"$\mu(c)$", r"$\mu(c) \exp(J(\pi, c))^{\eta}$",
                                                     r"$\argmin_{\mathcal{N}(\mu, \sigma)} D_{\text{KL}}(\mathcal{N}(\mu, \sigma) \| \mu(c))\ \text{s.t.}\ \mathbb{E}_{\mathcal{N}(\mu, \sigma)} \left[ J(\pi, c) \right] \geq \delta$"],
                                  ncol=4, fontsize=FONT_SIZE, loc='upper center', bbox_to_anchor=(2.87, 1.83),
                                  handlelength=0.4, columnspacing=0.5, handletextpad=0.3)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def kl_interpolation_plot2(path=None):
    xs = np.linspace(0, 1, 1000)
    target_lls1 = []
    target_lls2 = []
    alphass = [[0., 0.45, 0.5, 0.55, 1.], [0., 0.25, 0.5, 0.75, 1.]]

    target_ll1 = np.log(np.where(xs > 0.75, 1, 1e-5))
    target_ll1 -= logsumexp(target_ll1)
    target_lls1.append(target_ll1)

    target_ll2 = np.log(np.where(xs < 0.25, 1, 1e-5))
    target_ll2 -= logsumexp(target_ll2)
    target_lls2.append(target_ll2)

    target_ll1 = -500 * np.square(xs - 0.9)
    target_ll1 -= logsumexp(target_ll1)
    target_lls1.append(target_ll1)

    target_ll2 = -500 * np.square(xs - 0.1)
    target_ll2 -= logsumexp(target_ll2)
    target_lls2.append(target_ll2)

    # f = plt.figure(figsize=(6, 1.5))
    f = plt.figure(figsize=(6.3, 1.3))
    for row, (target_ll1, target_ll2, alphas) in enumerate(zip(target_lls1, target_lls2, alphass)):
        # Generate the different plots for the different eta
        axs = [f.add_axes([0.83 * i / len(alphas) + 0.005, 0.1 + 0.42 * row, 0.83 / len(alphas) - 0.008, 0.4]) for i in
               range(len(alphas))]
        # axs = [f.add_axes([i / len(alphas) + 0.015, 0.08 + 0.35 * row, 1 / len(alphas) - 0.03, 0.33]) for i in
        #        range(len(alphas))]
        for i, alpha in enumerate(alphas):
            interp_ll = target_ll1 * alpha + target_ll2 * (1 - alpha)
            interp_ll -= logsumexp(interp_ll)

            linit = np.exp(target_ll1) / (xs[1] - xs[0])
            linterp = np.exp(interp_ll) / (xs[1] - xs[0])
            ltarget = np.exp(target_ll2) / (xs[1] - xs[0])
            min_l = min(np.min(linterp), np.min(linit), np.min(ltarget))
            max_l = max(np.max(linterp), np.max(linit), np.max(ltarget))
            p1l, = axs[i].plot(xs, (linit - min_l) / (max_l - min_l), color="C0")
            axs[i].fill_between(xs, 0, (linit - min_l) / (max_l - min_l), color="C0", alpha=0.5)
            p2l, = axs[i].plot(xs, (ltarget - min_l) / (max_l - min_l), color="C1")
            axs[i].fill_between(xs, 0, (ltarget - min_l) / (max_l - min_l), color="C1", alpha=0.5)
            il, = axs[i].plot(xs, (linterp - min_l) / (max_l - min_l), color="C2")
            axs[i].fill_between(xs, 0, (linterp - min_l) / (max_l - min_l), color="C2", alpha=0.5)

            axs[i].set_xlim([0, 1])
            axs[i].set_xticks([0., 0.25, 0.5, 0.75, 1.])
            axs[i].set_ylim([0, 1])
            axs[i].set_yticks([0., 0.25, 0.5, 0.75, 1.])

            if row == 1:
                axs[i].set_title(r"$\alpha=%.2f$" % alpha, fontsize=FONT_SIZE, pad=0)

                if i == 0:
                    axs[i].legend([p1l, p2l, il],
                                  [r"$p_1(c)$", r"$p_2(c)$", r"$p_1(c)^{\alpha} p_2(c)^{1-\alpha}$"],
                                  ncol=1, fontsize=FONT_SIZE, loc='center right', bbox_to_anchor=(6.3, 0.),
                                  handlelength=0.4, columnspacing=0., handletextpad=0.3)

            if row == 0:
                axs[i].set_xlabel(r"$c$", fontsize=FONT_SIZE, labelpad=-1)

            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
            axs[i].grid()
            axs[i].tick_params('both', length=0, width=0, which='major')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def generate_histogram_samples(log_likelihood, bin_edges, n_samples):
    lls = log_likelihood(bin_edges[:-1] + 0.5 * (bin_edges[1:] - bin_edges[0:-1]))
    lls -= logsumexp(lls)

    hist = np.round(np.exp(lls) * n_samples).astype(np.int)
    samples = []
    for n_sub, start, end in zip(hist, bin_edges[:-1], bin_edges[1:]):
        samples.append(np.random.uniform(start, end, n_sub))

    return np.concatenate(samples, axis=0)[:, None]


def wasserstein_plot(path=None):
    np.random.seed(0)
    xs = np.linspace(0, 1, 1000)

    target_ll_fns = [lambda x: np.log(np.where(x > 0.76, 1, 1e-5)),
                     lambda x: -10 * np.square(x - 0.9)]
    cache_dirs = ["uniform_target_interpolations", "gauss_target_interpolations"]

    deltas = [0.95, 0.85, 0.3, 0.1, 0.]
    fig = plt.figure(figsize=(6, 1.3))
    for row, (target_ll_fn, cache_dir) in enumerate(zip(target_ll_fns, cache_dirs)):
        os.makedirs(cache_dir, exist_ok=True)
        target_ll = target_ll_fn(xs)
        target_ll -= logsumexp(target_ll)
        target_samples = generate_histogram_samples(target_ll_fn, np.linspace(0, 1, 51), 1000)
        n_samples = target_samples.shape[0]
        target_sampler = lambda n: target_samples
        vf = get_vf()
        values = np.squeeze(vf(torch.from_numpy(xs).float()[:, None]).detach().numpy())

        # We first pre-compute the results to avoid re-doing that when changing the format
        samples = {}
        for delta in deltas:
            cache_path = os.path.join(cache_dir, "interpolants_%.2f.pkl" % delta)
            if not os.path.exists(cache_path):
                init_samples = np.random.uniform(0., np.max(xs[values >= delta]), n_samples)[:, None]
                curriculum = WassersteinInterpolation(init_samples, target_sampler, perf_lb=delta, epsilon=10.,
                                                      opt_time=60., opt_tol=1e-6, ws_scaling=0.99, ws_blur=0.01)

                curriculum.current_samples = curriculum.current_samples - \
                                             np.random.uniform(0, 5e-3, curriculum.current_samples.shape)
                curriculum.update_distribution(vf, init_samples)

                samples[delta] = curriculum.current_samples.copy()
                with open(cache_path, "wb") as f:
                    pickle.dump(curriculum.current_samples, f)
            else:
                with open(cache_path, "rb") as f:
                    samples[delta] = pickle.load(f)

        # axs = [fig.add_axes([i / len(deltas) + 0.015, 0.08 + 0.35 * row, 1 / len(deltas) - 0.03, 0.33]) for i in
        #        range(len(deltas))]
        axs = [fig.add_axes([0.87 * i / len(deltas) + 0.005, 0.1 + 0.42 * row, 0.87 / len(deltas) - 0.01, 0.4]) for i in
               range(len(deltas))]
        for i, delta in enumerate(deltas):
            interp, bin_edges = np.histogram(np.squeeze(samples[delta]), bins=50, range=(0, 1))
            interp = interp / (n_samples * (bin_edges[1:] - bin_edges[:-1]))
            target = np.exp(target_ll) * (xs.shape[0] + 1)
            scale = max(np.max(target), interp[np.argsort(interp)[-2]])

            tl, = axs[i].plot(xs, target / scale, color="C0")
            axs[i].fill_between(xs, 0., target / scale, color="C0", alpha=0.5)
            il = axs[i].bar(bin_edges[:-1] + 0.5 * (bin_edges[1:] - bin_edges[:-1]),
                            interp / scale, width=bin_edges[1] - bin_edges[0], color="C2", alpha=0.9)

            vl, = axs[i].plot(xs, values, color="C1")
            axs[i].set_ylim([0, 1])
            axs[i].set_xlim([0, 1])
            axs[i].set_xticks([0., 0.25, 0.5, 0.75, 1.])
            axs[i].set_yticks([0., 0.25, 0.5, 0.75, 1.])
            axs[i].grid()

            axs[i].hlines(delta, 0., 1., color="black", linestyles="--")
            axs[i].fill_between(xs, 0., values, color="C1", alpha=0.3)
            if row == 1:
                if i == 0:
                    axs[i].legend([vl, tl, il], [r"$J(\pi, c)$", r"$\mu(c)$", r"$p_{\mathcal{W}}(c)$"],
                                  ncol=1, fontsize=FONT_SIZE, loc='center right', bbox_to_anchor=(5.9, 0.),
                                  handlelength=0.4, columnspacing=0., handletextpad=0.3)

                axs[i].set_title(r"$\delta=%.2f$" % delta, fontsize=FONT_SIZE, pad=0)
            if row == 0:
                axs[i].set_xlabel(r"$c$", fontsize=FONT_SIZE, labelpad=-1)

            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
            axs[i].tick_params('both', length=0, width=0, which='major')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    kl_interpolation_plot1(path="figures/value_interpolation.pdf")
    kl_interpolation_plot2(path="figures/kl_interpolation.pdf")
    wasserstein_plot(path="figures/wasserstein_interpolation.pdf")
