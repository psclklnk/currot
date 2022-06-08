import time
import numpy as np
from matplotlib import cm
import nadaraya_watson as na
import matplotlib.pyplot as plt


def target_fun(x):
    return np.sin(x[..., 0]) * x[..., 1]


np.random.seed(0)

data = np.random.uniform(-np.pi, np.pi, size=(1000, 3))
values = target_fun(data)
smoother = na.NadarayaWatson(data, values, n_threads=10)

test_data = np.random.uniform(-np.pi, np.pi, size=(1000, 3))
t1 = time.time()
for i in range(10):
    res = smoother.predict(test_data, bandwidth=1., n_max=300)
t2 = time.time()
print("Took: %.3e" % ((t2 - t1) / 10))

data = np.random.uniform(-np.pi, np.pi, size=(10000, 2))
values = target_fun(data)
smoother = na.NadarayaWatson(data, values, n_threads=10)

x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100))
z_gt = target_fun(np.reshape(np.stack((x, y), axis=-1), (-1, 2))).reshape(x.shape)
z = smoother.predict(np.reshape(np.stack((x, y), axis=-1), (-1, 2)), bandwidth=0.5, n_max=50).reshape(x.shape)

f, axs = plt.subplots(1, 2)
im = axs[0].imshow(z_gt, cmap=cm.RdBu, extent=[-np.pi, np.pi, -np.pi, np.pi], origin="lower")  # drawing the function
cset = axs[0].contour(x, y, z_gt, linewidths=2, cmap=cm.Set2, vmin=np.min(z_gt), vmax=np.max(z_gt))

im = axs[1].imshow(z, cmap=cm.RdBu, extent=[-np.pi, np.pi, -np.pi, np.pi], origin="lower")  # drawing the function
cset = axs[1].contour(x, y, z, linewidths=2, cmap=cm.Set2, vmin=np.min(z_gt), vmax=np.max(z_gt))
# axs[1].scatter(data[:, 0], data[:, 1], c=values, cmap=cm.RdBu)
plt.show()

if False:
    test_data = np.random.uniform(-3, 3, size=(50000, 3))

    n_runs = 50
    for n_max in [200, 100, 50, 20, 10, 5]:
        t1 = time.time()
        for i in range(0, n_runs):
            res = smoother.predict(test_data, bandwidth=0.5, n_max=n_max)
        t2 = time.time()
        print("N-Max: %d, Time: %.3e" % (n_max, (t2 - t1) / n_runs))
