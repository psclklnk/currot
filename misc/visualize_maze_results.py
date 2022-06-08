import os
import pickle
import matplotlib
import numpy as np
from matplotlib import cm
from misc.util import add_plot
import matplotlib.pyplot as plt
from deep_sprl.experiments import MazeExperiment

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
    "font.serif": "Helvetica",
    "font.family": "serif"
})

FONT_SIZE = 8
TICK_SIZE = 6


def maze():
    # Maze
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    # We draw a black white image
    maze_image = 255 * np.ones(maze.shape + (3,))
    x, y = np.where(maze == '1')
    maze_image[x, y, :] = 0.
    x, y = np.where(maze == "r")
    maze_image[x, y, 1:] = 0.

    return maze_image


def add_precision_plot(log_dir, ax, color):
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

            avg_precs = []
            for iteration in iterations:
                with open(os.path.join(seed_path, "iteration-%d" % iteration, "context_trace.pkl"), "rb") as f:
                    trace = pickle.load(f)
                if len(trace[0]) != 0:
                    avg_precs.append(np.median(np.array(trace[-1])[:, -1]))

            if len(avg_precs) < len(iterations):
                avg_precs = [avg_precs[0]] * (len(iterations) - len(avg_precs)) + avg_precs

            xs.append(iterations)
            ys.append(avg_precs)

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


def precision_comparison(ax=None, path=None, base_log_dir="logs"):
    if ax is None:
        f = plt.figure(figsize=(2.3, 1.4))
        ax = plt.Axes(f, [0.2, 0.23, 0.76, 0.52])
        f.add_axes(ax)
        show = True
    else:
        f = plt.gcf()
        show = False

    lines = []
    count = 0
    for method, color in zip(["self_paced", "random", "wasserstein", "goal_gan", "alp_gmm",
                              "acl", "plr", "vds"],
                             ["C0", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]):
        exp = MazeExperiment(base_log_dir, method, "sac", {}, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())

        try:
            lines.append(add_precision_plot(log_dir, ax, color))
        except Exception:
            lines.append(None)
        count += 1

    lines.append(ax.hlines(0.05, 0., 400, color="black", linestyle="--"))

    if show:
        f.legend(lines,
                 [r"\sprl", "Random", r"\currot", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds", "Min. Tol."],
                 fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(0.05, 1.01), ncol=4, columnspacing=0.4,
                 handlelength=0.6, handletextpad=0.25)

    ax.set_ylim([0.04, 18])
    ax.set_yscale("log")

    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xticklabels([r"$0$", r"$100$", r"$200$", r"$300$", r"$400$"])
    ax.set_xlim([0, 400])
    ax.set_xlabel(r"Epoch", fontsize=FONT_SIZE, labelpad=2)
    ax.set_ylabel(r"Tolerance", fontsize=FONT_SIZE, labelpad=1)

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    if show:
        if path is None:
            plt.show()
        else:
            plt.savefig(path)


def full_plot(path=None, base_log_dir="logs"):
    f = plt.figure(figsize=(4.83, 1.25))
    ax1 = f.add_axes([0.088, 0.24, 0.38, 0.56])
    ax2 = f.add_axes([0.6, 0.24, 0.38, 0.56])

    lines = performance_plot(ax1, base_log_dir=base_log_dir)
    precision_comparison(ax2, base_log_dir=base_log_dir)

    f.legend(lines,
             [r"\sprl", "Random", "Oracle", r"\currot \tiny{(Ours)}", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds"],
             fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(-0.01, 1.03), ncol=9, columnspacing=0.4,
             handlelength=0.9, handletextpad=0.25)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def performance_plot(ax=None, path=None, base_log_dir="logs"):
    if ax is None:
        f = plt.figure(figsize=(2.3, 1.4))
        ax = plt.Axes(f, [0.19, 0.23, 0.77, 0.52])
        f.add_axes(ax)
        show = True
    else:
        f = plt.gcf()
        show = False

    lines = []
    for method, color in zip(["self_paced", "random", "default", "wasserstein", "goal_gan", "alp_gmm",
                              "acl", "plr", "vds"],
                             ["C0", "C2", (0.2, 0.2, 0.2), "C4", "C5", "C6", "C7", "C8", "C9"]):
        exp = MazeExperiment(base_log_dir, method, "sac", {}, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))

    ax.set_ylabel(r"Success Rate", fontsize=FONT_SIZE, labelpad=2.)
    ax.set_xlabel(r"Epoch", fontsize=FONT_SIZE, labelpad=2.)

    if show:
        f.legend(lines,
                 [r"\sprl", "Random", "Oracle", r"\currot", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds"],
                 fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(0.02, 1.01), ncol=4, columnspacing=0.4,
                 handlelength=0.9, handletextpad=0.25)

    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xticklabels([r"$0$", r"$100$", r"$200$", r"$300$", r"$400$"])
    ax.set_xlim([0, 400])

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([r"$0$", r"$0.25$", r"$0.5$", r"$0.75$", r"$1$"])
    ax.set_ylim([0, 1])
    ax.grid()

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE)
    plt.grid()
    plt.tight_layout()
    if show:
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
    else:
        return lines


def wb_distribution_visualization(seed, iterations, path=None, base_log_dir="logs"):
    exp = MazeExperiment(base_log_dir, "wasserstein", "sac", {},
                         seed=0)
    log_dir = os.path.dirname(exp.get_log_dir())
    seed_path = os.path.join(log_dir, "seed-%d" % seed)

    norm = matplotlib.colors.Normalize(vmin=exp.LOWER_CONTEXT_BOUNDS[-1], vmax=0.2, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="viridis")

    teacher = exp.create_self_paced_teacher()
    # f = plt.figure(figsize=(1.5, 1.2))
    f = plt.figure(figsize=(1.5, 1.45))
    cax = f.add_axes([0.79, 0.05, 0.04, 0.9])
    for i, iteration in enumerate(iterations):
        teacher.load(os.path.join(seed_path, "iteration-%d" % iteration))
        samples = teacher.teacher.current_samples
        success_samples = teacher.success_buffer.contexts
        print((np.percentile(success_samples[:, -1], 25), np.percentile(success_samples[:, -1], 50),
               np.percentile(success_samples[:, -1], 75)))

        # ax = plt.Axes(f, [0.4 * (i % 2) + 0.005, 0.5 * (1 - (i // 2)) + 0.005, 0.39, 0.49])
        ax = plt.Axes(f, [0.39 * (i % 2) + 0.005, 0.5 * (1 - (i // 2)) + 0.005, 0.38, 0.42])
        f.add_axes(ax)

        ax.imshow(maze(), extent=(exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0], exp.LOWER_CONTEXT_BOUNDS[1],
                                  exp.UPPER_CONTEXT_BOUNDS[1]), origin="lower")
        scat = ax.scatter(success_samples[:, 0], success_samples[:, 1], alpha=0.3,
                          c=mapper.to_rgba(success_samples[:, 2]), s=2)

        ax.set_xlim(exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0])
        ax.set_ylim(exp.LOWER_CONTEXT_BOUNDS[1], exp.UPPER_CONTEXT_BOUNDS[1])

        ax.set_title(r"Epoch $%d$" % iteration, fontsize=TICK_SIZE, pad=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params('both', length=0, width=0, which='major')
        ax.set_rasterized(True)

        f.colorbar(scat, cax=cax, orientation='vertical')
        cax.tick_params('y', length=2, width=0.5, which='major', pad=1)
        cax.set_yticks([0.0, 0.33, 0.66, 1.0])
        cax.set_yticklabels([r"$0.05$", r"$0.1$", r"$0.15$", r"${\geq}0.2$"], fontsize=TICK_SIZE)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


def g_distribution_visualization(seed, iterations, path=None, base_log_dir="logs"):
    exp = MazeExperiment(base_log_dir, "self_paced", "sac", {}, seed=0)
    log_dir = os.path.dirname(exp.get_log_dir())
    seed_path = os.path.join(log_dir, "seed-%d" % seed)

    norm = matplotlib.colors.Normalize(vmin=exp.LOWER_CONTEXT_BOUNDS[-1], vmax=10., clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="viridis")

    teacher = exp.create_self_paced_teacher()
    # f = plt.figure(figsize=(1.55, 1.2))
    f = plt.figure(figsize=(1.5, 1.4))
    cax = f.add_axes([0.81, 0.05, 0.045, 0.9])
    for i, iteration in enumerate(iterations):
        teacher.load(os.path.join(seed_path, "iteration-%d" % iteration))
        samples = []
        for _ in range(0, 1000):
            samples.append(teacher.sample())
        samples = np.array(samples)

        # ax = plt.Axes(f, [0.4 * (i % 2) + 0.005, 0.5 * (1 - (i // 2)) + 0.005, 0.39, 0.49])
        ax = plt.Axes(f, [0.4 * (i % 2) + 0.005, 0.5 * (1 - (i // 2)) + 0.005, 0.39, 0.42])
        f.add_axes(ax)

        ax.imshow(maze(), extent=(exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0], exp.LOWER_CONTEXT_BOUNDS[1],
                                  exp.UPPER_CONTEXT_BOUNDS[1]), origin="lower")
        scat = ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, color=mapper.to_rgba(samples[:, 2]), s=2)

        ax.set_xlim(exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0])
        ax.set_ylim(exp.LOWER_CONTEXT_BOUNDS[1], exp.UPPER_CONTEXT_BOUNDS[1])

        ax.set_title(r"Epoch $%d$" % iteration, fontsize=TICK_SIZE, pad=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params('both', length=0, width=0, which='major')
        ax.set_rasterized(True)

        f.colorbar(scat, cax=cax, orientation='vertical')
        cax.tick_params('y', length=2, width=0.5, which='major', pad=1)
        cax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cax.set_yticklabels([r"$0.05$", r"$2$", r"$4$", r"$6$", r"$8$", r"${\geq}10$"], fontsize=TICK_SIZE)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


def g_variance_plot(ax=None, path=None, base_log_dir="logs"):
    if ax is None:
        f = plt.figure(figsize=(2.3, 1.25))
        ax = plt.Axes(f, [0.18, 0.24, 0.76, 0.56])
        f.add_axes(ax)
        show = True
    else:
        f = plt.gcf()
        show = False

    exp = MazeExperiment(base_log_dir, "self_paced", "sac", {}, seed=0)
    teacher = exp.create_self_paced_teacher()
    log_dir = os.path.dirname(exp.get_log_dir())

    ys = []
    lines = []
    if os.path.exists(log_dir):
        seed_dirs = [f for f in os.listdir(log_dir) if f.startswith("seed")]
        for seed_dir in seed_dirs:
            seed_path = os.path.join(log_dir, seed_dir)
            iteration_dirs = [d for d in os.listdir(seed_path) if d.startswith("iteration-")]
            unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
            idxs = np.argsort(unsorted_iterations)
            iterations = unsorted_iterations[idxs]

            stds = []
            for iteration in iterations:
                teacher.load(os.path.join(seed_path, "iteration-%d" % iteration))
                stds.append(np.sqrt(np.diag(teacher.context_dist.covariance_matrix())))

            ys.append(stds)

    if len(ys) > 0:
        print("Found %d completed seeds" % len(ys))
        min_length = np.min([len(y) for y in ys])
        iterations = iterations[0: min_length]
        ys = [y[0: min_length] for y in ys]

        low, mid, high = np.percentile(ys, [10, 50, 90], axis=0)

        for dim in range(3):
            lines.append(ax.plot(iterations, mid[:, dim], color="C%d" % dim, linewidth=1)[0])
            ax.fill_between(iterations, low[:, dim], high[:, dim], color="C%d" % dim, alpha=0.5)

    if show:
        f.legend(lines, ["$x$-Position", "$y$-Position", "Tolerance"],
                 fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(0.09, 1.03), ncol=4, columnspacing=0.4,
                 handlelength=0.6, handletextpad=0.25)

    ax.set_yscale("log")
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xticklabels([r"$0$", r"$100$", r"$200$", r"$300$", r"$400$"])
    ax.set_xlim([0, 400])
    ax.set_xlabel(r"Epoch", fontsize=FONT_SIZE, labelpad=2)
    ax.set_ylabel(r"Standard Deviation", fontsize=FONT_SIZE, labelpad=1)
    ax.grid()

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    if show:
        if path is None:
            plt.show()
        else:
            plt.savefig(path)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs")
    full_plot(path="figures/maze_precision+performance.pdf", base_log_dir=base_log_dir)
    wb_distribution_visualization(1, [10, 60, 110, 300], path="figures/maze_distribution.pdf",
                                  base_log_dir=base_log_dir)
    g_variance_plot(path="figures/maze_sprl_variances.pdf", base_log_dir=base_log_dir)
    g_distribution_visualization(1, [10, 60, 110, 300], path="figures/maze_sprl_distribution.pdf",
                                 base_log_dir=base_log_dir)
