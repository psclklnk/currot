import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def add_plot(base_log_dir, ax, color, max_seed=None, marker="o", markevery=3):
    iterations = []
    seed_performances = []

    if max_seed is None:
        max_seed = int(1e6)

    if not os.path.exists(base_log_dir):
        return None

    seeds = [int(d.split("-")[1]) for d in os.listdir(base_log_dir) if d.startswith("seed-")]
    for seed in [s for s in seeds if s <= max_seed]:
        seed_dir = "seed-" + str(seed)
        seed_log_dir = os.path.join(base_log_dir, seed_dir)

        if os.path.exists(os.path.join(seed_log_dir, "performance.pkl")):
            iteration_dirs = [d for d in os.listdir(seed_log_dir) if d.startswith("iteration-")]
            unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
            idxs = np.argsort(unsorted_iterations)
            iterations = unsorted_iterations[idxs]

            with open(os.path.join(seed_log_dir, "performance.pkl"), "rb") as f:
                seed_performances.append(pickle.load(f))
        else:
            pass
            # raise RuntimeError("No Performance log was found")

    if len(seed_performances) > 0:
        print("Found %d completed seeds" % len(seed_performances))
        min_length = np.min([len(seed_performance) for seed_performance in seed_performances])
        iterations = iterations[0: min_length]
        seed_performances = [seed_performance[0: min_length] for seed_performance in seed_performances]

        mid = np.mean(seed_performances, axis=0)
        sem = np.std(seed_performances, axis=0) / np.sqrt(len(seed_performances))
        low = mid - 2 * sem
        high = mid + 2 * sem

        l, = ax.plot(iterations, mid, color=color, linewidth=1, marker=marker, markersize=2, markevery=markevery)
        ax.fill_between(iterations, low, high, color=color, alpha=0.5)
        return l
    else:
        return None


def get_auc(base_log_dir):
    seed_performances = []

    if not os.path.exists(base_log_dir):
        raise RuntimeError("No performance logs available")

    seeds = [int(d.split("-")[1]) for d in os.listdir(base_log_dir) if d.startswith("seed-")]
    for seed in seeds:
        seed_dir = "seed-" + str(seed)
        seed_log_dir = os.path.join(base_log_dir, seed_dir)

        if os.path.exists(os.path.join(seed_log_dir, "performance.pkl")):
            with open(os.path.join(seed_log_dir, "performance.pkl"), "rb") as f:
                seed_performances.append(np.sum(pickle.load(f)))
        else:
            raise RuntimeError("No Performance log was found")

    return np.mean(seed_performances)


def select_best_hps(experiment, learner, teacher, params, log_dir):
    full_params = [{"hard_likelihood": False}]
    for key, values in params.items():
        new_full_params = []
        for value in values:
            for pref in full_params:
                new_full_params.append({**pref, **{key: value}})
        full_params = new_full_params

    base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", log_dir)
    performances = []
    for full_param in full_params:
        exp = experiment(base_log_dir, teacher, learner, full_param.copy(), seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())
        performances.append(get_auc(log_dir))

    best_idxs = np.argsort(performances)[-5:]
    return [full_params[bi] for bi in best_idxs]


def param_comp(experiment, learner="sac", teacher="acl", params=None, full_params=None, log_dir="logs",
               font_size=None):
    base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", log_dir)

    if full_params is None:
        if params is None:
            raise RuntimeError("Either params or full_params need to be filled")

        full_params = [{"hard_likelihood": False}]
        for key, values in params.items():
            new_full_params = []
            for value in values:
                for pref in full_params:
                    new_full_params.append({**pref, **{key: value}})
            full_params = new_full_params

    f = plt.figure()
    ax = f.gca()

    colors = ["C0", "C1", "C2", "C4", "C5", "C6"] * int(np.ceil(len(full_params) / 6))
    lines = []
    labels = []
    for color, full_param in zip(colors, full_params):
        exp = experiment(base_log_dir, teacher, learner, full_param, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))
        labels.append(",".join("%s:%s" % (k, v) for k, v in full_param.items()))

    plt.xlabel("Epoch", fontsize=font_size, labelpad=2.)
    f.legend(lines, labels, fontsize=font_size, columnspacing=0.6, handlelength=1.0, loc="lower left")

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)
    plt.grid()
    plt.tight_layout()
    plt.show()
