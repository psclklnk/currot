import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from deep_sprl.teachers.util import NadarayaWatsonPy, NadarayaWatson


def fn1(x):
    return np.exp(0.2 * x[..., 1]) * np.sin(x[..., 0])


def main(est_class, check_gradients=False, **est_args):
    X = np.stack(np.meshgrid(np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)), axis=-1)
    Z = fn1(X)

    f, axs = plt.subplots(1, 2)
    im = axs[0].imshow(Z, cmap=cm.RdBu, extent=[-np.pi, np.pi, -np.pi, np.pi], origin="lower")  # drawing the function
    cset = axs[0].contour(X[..., 0], X[..., 1], Z, linewidths=2, cmap=cm.Set2)

    x = np.random.uniform(-np.pi, np.pi, size=(500, 2))
    y = fn1(x)

    pred = est_class(x, y, **est_args)
    Z_pred = pred.predict_individual(X)
    im = axs[1].imshow(Z_pred, cmap=cm.RdBu, extent=[-np.pi, np.pi, -np.pi, np.pi], origin="lower",
                       alpha=0.8)  # drawing the function
    cset = axs[1].contour(X[..., 0], X[..., 1], Z_pred, linewidths=2, cmap=cm.Set2)
    axs[1].scatter(x[:, 0], x[:, 1], c=y, cmap=cm.RdBu)
    plt.show()

    if check_gradients:
        # Do a finite difference check of the gradients
        xs_test = np.random.uniform(-np.pi, np.pi, size=(1000, 2))
        grads = pred.predict_individual(xs_test, with_gradient=True)[1]
        for grad, x_test in zip(grads, xs_test):
            grad_est = []
            for dim in range(0, 2):
                x_test[dim] += 1e-5
                pred_hi = pred.predict_individual(x_test)
                x_test[dim] -= 2e-5
                pred_lo = pred.predict_individual(x_test)
                x_test[dim] += 1e-5
                grad_est.append((pred_hi - pred_lo) / 2e-5)
            grad_est = np.array(grad_est)

            rel_err = np.linalg.norm(grad - grad_est) / max(1., np.linalg.norm(grad_est))
            assert rel_err < 1e-3


def comparison():
    x = np.random.uniform(-np.pi, np.pi, size=(30, 2))
    y = fn1(x)

    ref = NadarayaWatsonPy(x, y, lengthscale=2)
    new = NadarayaWatson(x, y, lengthscale=2, n_max=100, radius_scale=10.)

    for i in range(0, 10):
        x_test = np.random.uniform(-np.pi, np.pi, size=(2,))
        pred_new = np.squeeze(new.predict_individual(x_test.copy()))
        pred_ref = ref.predict_individual(x_test.copy())

        assert np.isclose(pred_ref, pred_new)


if __name__ == "__main__":
    comparison()

    for lengthscale in [2., 1., 0.5, 0.25, 0.125, 0.05, 0.025, 0.01, 0.001, 1e-6]:
        print("Lengthscale: %.3e" % lengthscale)
        np.random.seed(0)
        main(NadarayaWatsonPy, check_gradients=False, lengthscale=lengthscale)

        np.random.seed(0)
        main(NadarayaWatson, check_gradients=False, lengthscale=lengthscale, n_max=int(1e6), radius_scale=10.)
