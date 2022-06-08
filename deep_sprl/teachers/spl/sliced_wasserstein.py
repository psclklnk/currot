import numpy as np
import matplotlib.pyplot as plt


def sliced_wasserstein(source, target, n=100, grad=False):
    # Generate random projection vectors
    dim = source.shape[1]
    directions = np.random.normal(0, 1, (n, dim))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    # Compute the projected assignments
    source_projections = np.einsum("nd,md->nm", directions, source)
    target_projections = np.einsum("nd,md->nm", directions, target)

    sorted_source = np.argsort(source_projections, axis=-1)
    reverse_source = np.zeros_like(sorted_source)
    reverse_source[np.arange(0, n)[:, None], sorted_source] = np.arange(0, source.shape[0])[None, :]
    sorted_target = np.argsort(target_projections, axis=-1)

    proj_diffs = target_projections[np.arange(n)[:, None], sorted_target[np.arange(n)[:, None], reverse_source]] - \
                 source_projections
    swd = np.mean(np.square(proj_diffs))

    if grad:
        return swd, np.einsum("ij,id->jd", proj_diffs, directions) / n
    else:
        return swd


if __name__ == "__main__":
    source = np.random.multivariate_normal(np.ones(2), 0.2 * np.eye(2), size=200)
    target = np.random.multivariate_normal(5 * np.ones(2), 0.2 * np.eye(2), size=200)

    plt.scatter(source[:, 0], source[:, 1])
    plt.scatter(target[:, 0], target[:, 1])
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.show()

    for i in range(0, 100):
        swd, grad = sliced_wasserstein(source, target, grad=True, n=100)
        print("%.3e" % swd)
        source += grad

    plt.scatter(source[:, 0], source[:, 1])
    plt.scatter(target[:, 0], target[:, 1])
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.show()
