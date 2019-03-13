import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.reset_orig()

def plot_psds(x, y, z, dz = None, chs = 0, ax = None):

    if ax is None:
        ax = plt.gca()

    n   = y.shape[1]
    pal = sns.cubehelix_palette(n, start = chs)

    if dz is None:
        dz = np.diff(z).mean() / 8.0

    for i in range(n - 1, 0, -1):
        y_0 = z[i]
        y_1 = np.maximum(np.log10(y[:, i]), 0) * dz + y_0
        ax.plot(x, y_1, c = "white", zorder = -i)
        ax.fill_between(x, y_1, y_0, color = pal[i], zorder = -i)
        ax.plot(x, y_0 * np.ones(x.size), c = pal[i], zorder = -i)

    ax.set_xscale("log")
    ax.grid(False)
