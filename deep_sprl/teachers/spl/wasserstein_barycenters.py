import os
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from deep_sprl.teachers.spl.sliced_wasserstein import sliced_wasserstein


class RandomizedIndividualBarycenterCurriculum:

    def __init__(self, init_samples, target_sampler, perf_lb, eta, bounds, callback=None):
        self.current_samples = init_samples
        self.n_samples, self.dim = self.current_samples.shape
        self.target_sampler = target_sampler
        self.bounds = bounds
        self.perf_lb = perf_lb
        self.eta = eta
        self.callback = callback

    def sample_ball(self, targets, samples=None, half_ball=None, n=100):
        if samples is None:
            samples = self.current_samples

        # Taken from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        # Method 20
        direction = np.random.normal(0, 1, (n, self.dim))
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        r = np.power(np.random.uniform(size=(n, 1)), 1. / self.dim)

        # We only consider samples that decrease the distance objective (i.e. are aligned with the direction)
        noise = r * (direction / norm)
        dirs = targets - samples
        dir_norms = np.einsum("ij,ij->i", dirs, dirs)
        noise_projections = np.einsum("ij,kj->ik", dirs / dir_norms[:, None], noise)

        projected_noise = np.where((noise_projections > 0)[..., None], noise[None, ...],
                                   noise[None, ...] - 2 * noise_projections[..., None] * dirs[:, None, :])
        if half_ball is not None:
            projected_noise[~half_ball] = noise

        scales = np.minimum(self.eta, np.sqrt(dir_norms))[:, None, None]
        return np.squeeze(np.clip(samples[..., None, :] + scales * projected_noise, self.bounds[0], self.bounds[1]))

    @staticmethod
    def visualize_particles(init_samples, particles, performances):
        if particles.shape[-1] != 2:
            raise RuntimeError("Can only visualize 2D data")

        import matplotlib.pyplot as plt
        f = plt.figure()
        ax = f.gca()
        scat = ax.scatter(particles[0, :, 0], particles[0, :, 1], c=performances[0, :])
        ax.scatter(init_samples[0, 0], init_samples[0, 1], marker="x", c="red")
        plt.colorbar(scat)
        plt.show()

    def ensure_successful_initial(self, model, init_samples, success_samples):
        performance_reached = model.predict_individual(init_samples) >= self.perf_lb
        replacement_mask = ~performance_reached
        n_replacements = np.sum(replacement_mask)
        if n_replacements > 0:
            valid_successes = model.predict_individual(success_samples) >= self.perf_lb
            n_valid = np.sum(valid_successes)
            if n_valid >= n_replacements:
                # In this case we only allow selection from the valid samples
                success_samples = success_samples[valid_successes, :]
                valid_successes = np.ones(success_samples.shape[0], dtype=np.bool)

            dists = np.sum(np.square(success_samples[None, :, :] - init_samples[replacement_mask, None, :]), axis=-1)
            success_assignment = linear_sum_assignment(dists, maximize=False)
            init_samples[replacement_mask, :] = success_samples[success_assignment[1], :]
            performance_reached[replacement_mask] = valid_successes[success_assignment[1]]

        return init_samples, performance_reached

    def update_distribution(self, model, success_samples, debug=False):
        init_samples, performance_reached = self.ensure_successful_initial(model, self.current_samples.copy(),
                                                                           success_samples)
        target_samples = self.target_sampler(self.n_samples)
        if debug:
            target_samples_true = target_samples.copy()
        movements = sliced_wasserstein(init_samples, target_samples, grad=True)[1]
        target_samples = init_samples + movements
        particles = self.sample_ball(target_samples, samples=init_samples, half_ball=performance_reached)

        distances = np.linalg.norm(particles - target_samples[:, None, :], axis=-1)
        performances = model.predict_individual(particles)
        if debug:
            self.visualize_particles(init_samples, particles, performances)

        mask = performances > self.perf_lb
        solution_possible = np.any(mask, axis=-1)
        distances[~mask] = np.inf
        opt_idxs = np.where(solution_possible, np.argmin(distances, axis=-1), np.argmax(performances, axis=-1))
        new_samples = particles[np.arange(0, self.n_samples), opt_idxs]

        if debug:
            vis_idxs = np.random.randint(0, target_samples.shape[0], size=50)
            import matplotlib.pyplot as plt
            xs, ys = np.meshgrid(np.linspace(0, 9, num=150), np.linspace(0, 6, num=100))
            zs = model.predict_individual(np.stack((xs, ys), axis=-1))
            ims = plt.imshow(zs, extent=[0, 9, 0, 6], origin="lower")
            plt.contour(xs, ys, zs, [180])
            plt.colorbar(ims)

            plt.scatter(target_samples_true[vis_idxs, 0], target_samples_true[vis_idxs, 1], marker="x", color="red")
            plt.scatter(self.current_samples[vis_idxs, 0], self.current_samples[vis_idxs, 1], marker="o", color="C0")
            plt.scatter(init_samples[vis_idxs, 0], init_samples[vis_idxs, 1], marker="o", color="C2")
            plt.scatter(new_samples[vis_idxs, 0], new_samples[vis_idxs, 1], marker="o", color="C1")
            plt.xlim([0, 9])
            plt.ylim([0, 6])
            plt.show()

        if self.callback is not None:
            self.callback(self.current_samples, new_samples, success_samples, target_samples)

        self.current_samples = new_samples

    def save(self, path):
        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump((self.current_samples, self.perf_lb, self.eta), f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            tmp = pickle.load(f)

            self.current_samples = tmp[0]
            self.n_samples = self.current_samples.shape[0]

            self.perf_lb = tmp[1]
            self.eta = tmp[2]
