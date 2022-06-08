import os
import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from deep_sprl.experiments import PointMass2DExperiment, CurriculumType
from misc.visualize_point_mass_results import currot_plot, sprl_plot

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\usepackage{amssymb}')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})

FONT_SIZE = 8
TICK_SIZE = 6


def plot_trajectories(seed, path=None, base_log_dir="logs"):
    eval_iter = 195
    # Create the evaluation environment
    exp = PointMass2DExperiment(base_log_dir, "default", "ppo", {}, seed)
    types = ["default", "random", "self_paced", "wasserstein", "alp_gmm", "goal_gan", "plr", "vds", "acl"]
    for cur_type in types:
        f = plt.figure(figsize=(1., 1.))
        ax = plt.Axes(f, [0., 0., 1., 1.])
        f.add_axes(ax)

        width = 0.35
        ax.plot([-5., -3 - width], [-0.1, -0.1], linewidth=3, color="black")
        ax.plot([-3 + width, 3 - width], [-0.1, -0.1], linewidth=3, color="black")
        ax.plot([3 + width, 5.], [-0.1, -0.1], linewidth=3, color="black")

        ax.plot([-0.25, 0.25], [-3.25, -2.75], linewidth=3, color="red")
        ax.plot([-0.25, 0.25], [-2.75, -3.25], linewidth=3, color="red")

        exp.curriculum = CurriculumType.from_string(cur_type)
        iter_path = os.path.join(os.path.dirname(__file__), "..", exp.get_log_dir(), "iteration-%d" % eval_iter)

        if os.path.exists(iter_path):
            model = exp.learner.load_for_evaluation(os.path.join(iter_path, "model"), exp.vec_eval_env)

            np.random.seed(10)
            for i in range(0, 20):
                done = False
                obs = exp.vec_eval_env.reset()

                if obs[0][-2] < 0:
                    color = "C0"
                else:
                    color = "red"

                trajectory = [obs[0][[0, 2]]]
                while not done:
                    action = model.step(obs, state=None, deterministic=False)
                    obs, reward, done, info = exp.vec_eval_env.step(action)

                    # We need to add this check because the vectorized environment automatically resets everything on
                    # done
                    if not done:
                        trajectory.append(obs[0][[0, 2]])

                trajectory = np.array(trajectory)

                ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.2, linewidth=2)

        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_xticks([])
        ax.set_yticks([])

        if path is None:
            plt.show()
        else:
            plt.savefig(path + ("_%s.pdf" % cur_type))


def draw_env(ax, pos, width):
    ax.plot([-5., pos - (0.5 * width + 0.5)], [-0.1, -0.1], linewidth=3, color="black")
    ax.plot([pos + (0.5 + 0.5 * width), 5.], [-0.1, -0.1], linewidth=3, color="black")

    ax.scatter([0.], [3.], s=10, color="black")
    ax.plot([-0.25, 0.25], [-3.25, -2.75], linewidth=2, color="red")
    ax.plot([-0.25, 0.25], [-2.75, -3.25], linewidth=2, color="red")
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params('both', length=0, width=0, which='major')


def context_space_image(path=None, sprl=False, base_log_dir="logs"):
    f = plt.figure(figsize=(2.0, 1.4))

    ax = plt.Axes(f, [0.178, 0.08, 0.6435, 0.85])
    f.add_axes(ax)
    ax.set_xlim(-4., 4.)
    ax.set_ylim(0., 8.2)

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position(('data', 0.5))
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create custom major ticks to determine position of tick labels
    ax.set_xticks([-3, 3])
    ax.set_xticklabels([r'$-3$', r'$3$'], fontsize=TICK_SIZE)
    ax.set_yticks([4])
    ax.set_yticklabels([r'$4$'], fontsize=TICK_SIZE)

    ax.set_xticks(np.arange(-3, 4), minor=True)
    ax.set_yticks(np.arange(0., 8), minor=True)

    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0.5), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

    colormap = cm.get_cmap('viridis', 256)
    if sprl:
        sprl_plot(1, [10, 25, 35, 50], axs=[ax, ax, ax, ax],
                    colors=[colormap(0.1), colormap(0.3), colormap(0.5), colormap(1.)], s=30,
                    base_log_dir=base_log_dir)
    else:
        currot_plot(1, [10, 30, 50, 100], axs=[ax, ax, ax, ax],
                     colors=[colormap(0.1), colormap(0.3), colormap(0.5), colormap(1.)], s=30,
                     base_log_dir=base_log_dir)

    ax.scatter([-3, 3, 0], [0.5, 0.5, 5], s=20, c='black', marker='x')
    ax.text(-5.9, 7.7, r'$\mathcal{C} \subseteq \mathbb{R}^2$', fontsize=FONT_SIZE)

    ax = plt.Axes(f, [0.03, 0.25, 0.21, 0.3])
    f.add_axes(ax)
    draw_env(ax, -3, 0.8)

    ax = plt.Axes(f, [0.78, 0.2, 0.21, 0.3])
    f.add_axes(ax)
    draw_env(ax, 3, 0.8)

    ax = plt.Axes(f, [0.74, 0.66, 0.21, 0.3])
    f.add_axes(ax)
    draw_env(ax, 0., 5.)

    ax.set_xlabel(r'$w_g$', size=FONT_SIZE, labelpad=55, x=0.75)
    ax.set_ylabel(r'$p_g$', size=FONT_SIZE, labelpad=37, y=0.9, rotation=0)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


if __name__ == "__main__":
    base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs")
    os.makedirs("figures", exist_ok=True)
    context_space_image("figures/point_mass_env.pdf", sprl=False, base_log_dir=base_log_dir)
    context_space_image("figures/point_mass_distribution_sprl.pdf", sprl=True, base_log_dir=base_log_dir)
    plot_trajectories(1, path="figures/point_mass_trajectories", base_log_dir=base_log_dir)
