import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from deep_sprl.experiments import PointMass2DExperiment
from misc.util import add_plot

plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\newcommand{\currot}{\textsc{currot}}'
                              r'\newcommand{\sprl}{\textsc{sprl}}'
                              r'\newcommand{\alpgmm}{\textsc{alp-gmm}}'
                              r'\newcommand{\goalgan}{\textsc{goalgan}}'
                              r'\newcommand{\acl}{\textsc{acl}}'
                              r'\newcommand{\vds}{\textsc{vds}}'
                              r'\newcommand{\plr}{\textsc{plr}}')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})

FONT_SIZE = 8
TICK_SIZE = 6


def performance_plot(ax, base_log_dir="logs"):
    lines = []
    for method, color in zip(["self_paced", "random", "default", "wasserstein"],
                             ["C0", "C2", "C3", "C4"]):
        exp = PointMass2DExperiment(base_log_dir, method, "ppo", {}, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))

    # Add the other baselines (always in the hard environment)
    for method, color in zip(["goal_gan", "alp_gmm", "acl", "plr", "vds"], ["C5", "C6", "C7", "C8", "C9"]):
        exp = PointMass2DExperiment(base_log_dir, method, "ppo", {}, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))

    ax.set_yticks([3, 6])
    ax.set_yticklabels([3, 6])
    ax.set_xlim([0, 200])

    return lines


def sprl_plot(seed, iterations, path=None, axs=None, colors=None, s=10, base_log_dir="logs"):
    exp = PointMass2DExperiment(base_log_dir, "self_paced", "ppo", {}, seed=seed)
    seed_path = exp.get_log_dir()

    teacher = exp.create_self_paced_teacher()
    if axs is None:
        show = True
        f = plt.figure(figsize=(4.1, 0.5))
        assert len(iterations) == 4
        axs = [f.add_axes([0.25 * i + 0.01, 0, 0.23, 1]) for i in range(4)]
    else:
        show = False

    if colors is None:
        colors = ["C0"] * 4

    for i, iteration in enumerate(iterations):
        teacher.load(os.path.join(seed_path, "iteration-%d" % iteration))
        ax = axs[i]

        samples = []
        for _ in range(0, 1000):
            samples.append(teacher.sample())
        samples = np.array(samples)

        ax.scatter(samples[:, 0], samples[:, 1], s=s, alpha=0.2, color=colors[i])

        if show:
            ax.set_xlim(exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0])
            ax.set_ylim(exp.LOWER_CONTEXT_BOUNDS[1], 0.5 * exp.UPPER_CONTEXT_BOUNDS[1])
            ax.set_xticks([-3., -2., -1., 0., 1., 2., 3.])
            ax.set_yticks([1.125, 1.875, 2.625, 3.375])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params('both', length=0, width=0, which='major')
            ax.set_axisbelow(True)
            ax.grid()

    if show:
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()


def currot_plot(seed, iterations, axs=None, path=None, colors=None, s=10, base_log_dir="logs"):
    exp = PointMass2DExperiment(base_log_dir, "wasserstein", "ppo", {}, seed=0)
    log_dir = os.path.dirname(exp.get_log_dir())
    seed_path = os.path.join(log_dir, "seed-%d" % seed)

    show = False
    if axs is None:
        show = True
        f = plt.figure(figsize=(1.4, 1.4))
        assert len(iterations) == 4
        axs = [f.add_axes([0.5 * (i % 2) + 0.005, 0.5 * (1 - (i // 2)) + 0.005, 0.48, 0.48]) for i in range(4)]

    if colors is None:
        colors = ["C4"] * 4

    teacher = exp.create_self_paced_teacher()
    for i, iteration in enumerate(iterations):
        ax = axs[i]
        teacher.load(os.path.join(seed_path, "iteration-%d" % iteration))

        ax.scatter(teacher.teacher.current_samples[:, 0], teacher.teacher.current_samples[:, 1], s=s, alpha=0.2,
                   color=colors[i])

        if show:
            ax.set_xlim(exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0])
            ax.set_ylim(exp.LOWER_CONTEXT_BOUNDS[1], exp.UPPER_CONTEXT_BOUNDS[1])
            ax.set_xticks([-3., -2., -1., 0., 1., 2., 3.])
            ax.set_yticks([2., 3.5, 5, 6.5])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params('both', length=0, width=0, which='major')
            ax.set_axisbelow(True)
            ax.grid()

    if show:
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()


def full_performance_plot(ax=None, path=None, base_log_dir="logs"):
    if ax is None:
        f = plt.figure(figsize=(2.3, 1.4))
        ax = plt.Axes(f, [0.13, 0.23, 0.85, 0.55])
        f.add_axes(ax)
        show = True
    else:
        f = plt.gcf()
        show = False

    lines = performance_plot(ax, base_log_dir=base_log_dir)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE)
    ax.set_xlabel("Epoch", fontsize=FONT_SIZE, labelpad=2)
    ax.grid()
    ax.set_ylabel("Cum. Disc. Ret.", fontsize=FONT_SIZE, labelpad=2)

    if show:
        f.legend(lines,
                 [r"\sprl", "Random", "Default", r"\currot", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds"],
                 fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(-0.005, 1.015), ncol=4, columnspacing=0.6,
                 handlelength=1.0, handletextpad=0.3)

        if path is None:
            plt.show()
        else:
            plt.savefig(path)
    else:
        return lines


def add_precision_plot(log_dir, ax, color, dim):
    xs = []
    ys = []

    if os.path.exists(log_dir):
        seed_dirs = [f for f in os.listdir(log_dir) if f.startswith("seed")]
        for seed_dir in seed_dirs:
            seed_path = os.path.join(log_dir, seed_dir)
            iteration_dirs = [d for d in os.listdir(seed_path) if d.startswith("iteration-")]
            unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
            idxs = np.argsort(unsorted_iterations)
            iterations = unsorted_iterations[idxs]

            avg_dists = []
            for iteration in iterations:
                with open(os.path.join(seed_path, "iteration-%d" % iteration, "context_trace.pkl"), "rb") as f:
                    trace = pickle.load(f)

                if len(trace[0]) != 0:
                    contexts = np.array(trace[-1])
                    if dim == 0:
                        min_width_dist = np.minimum(np.abs(contexts[:, 0] - (-3)), np.abs(contexts[:, 0] - 3))
                    else:
                        min_width_dist = np.abs(contexts[:, 1] - 0.5)
                    avg_dists.append(np.median(min_width_dist))

            if len(avg_dists) < len(iterations):
                avg_dists = [avg_dists[0]] * (len(iterations) - len(avg_dists)) + avg_dists

            xs.append(iterations)
            ys.append(avg_dists)

    if len(ys) > 0:
        print("Found %d completed seeds" % len(ys))
        min_length = np.min([len(y) for y in ys])
        iterations = iterations[0: min_length]
        ys = [y[0: min_length] for y in ys]

        mid = np.mean(ys, axis=0)
        sem = np.std(ys, axis=0) / np.sqrt(len(ys))
        low = mid - 2 * sem
        high = mid + 2 * sem

        l, = ax.plot(iterations, mid, color=color, linewidth=1)
        ax.fill_between(iterations, low, high, color=color, alpha=0.5)
        return l
    else:
        return None


def precision_comparison(axs=None, path=None, base_log_dir="logs"):
    if axs is None:
        f = plt.figure(figsize=(3., 1.4))
        axs = []
        axs.append(f.add_axes([0.1, 0.23, 0.4, 0.52]))
        axs.append(f.add_axes([0.6, 0.23, 0.4, 0.52]))
        show = True
    else:
        f = plt.gcf()
        show = False

    count = 0
    for i in range(0, 2):
        lines = []
        for method, color in zip(["self_paced", "random", "wasserstein", "goal_gan", "alp_gmm",
                                  "acl", "plr", "vds"],
                                 ["C0", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]):
            exp = PointMass2DExperiment(base_log_dir, method, "ppo", {}, seed=0)
            log_dir = os.path.dirname(exp.get_log_dir())

            try:
                lines.append(add_precision_plot(log_dir, axs[i], color, dim=i))
            except Exception:
                lines.append(None)
            count += 1

            # axs[i].set_ylim([0., axs[i].get_ylim()[1]])

        axs[i].set_xticks([0, 50, 100, 150, 200])
        axs[i].set_xticklabels([r"$0$", r"$50$", r"$100$", r"$150$", r"$200$"])
        axs[i].set_xlim([0, 200])
        axs[i].set_xlabel("Epoch", fontsize=FONT_SIZE, labelpad=2)
        axs[i].set_ylabel("Gate Position" if i == 0 else "Gate Width", fontsize=FONT_SIZE, labelpad=1)
        axs[i].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        axs[i].grid()

    if show:
        f.legend(lines,
                 [r"\sprl", "Random", r"\currot", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds"],
                 fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(0.05, 1.01), ncol=6, columnspacing=0.4,
                 handlelength=0.6, handletextpad=0.25)

        if path is None:
            plt.show()
        else:
            plt.savefig(path)
    else:
        return lines


def combined_plot(path=None, base_log_dir="logs"):
    f = plt.figure(figsize=(6.0, 1.25))
    ax1 = f.add_axes([0.07, 0.24, 0.27, 0.56])
    ax2 = f.add_axes([0.43, 0.24, 0.23, 0.56])
    ax3 = f.add_axes([0.74, 0.24, 0.23, 0.56])

    lines = full_performance_plot(ax1, base_log_dir=base_log_dir)
    precision_comparison([ax2, ax3], base_log_dir=base_log_dir)

    f.legend(lines,
             [r"\sprl", "Random", "Default", r"\currot \tiny{(Ours)}", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds"],
             fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(0.095, 1.03), ncol=9, columnspacing=0.6,
             handlelength=0.9, handletextpad=0.25)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs")
    combined_plot(path="figures/point_mass_precision+performance.pdf", base_log_dir=base_log_dir)
