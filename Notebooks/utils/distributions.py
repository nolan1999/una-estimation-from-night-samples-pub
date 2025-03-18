import numpy as np
import scipy
import statsmodels.api as sm


def density_plot(data, ax, max_, **plot_keywords):
    density = scipy.stats.gaussian_kde(data)
    xs = np.linspace(0, max_, 200)
    density.covariance_factor = lambda: 0.2
    density._compute_covariance()
    ax.plot(xs, density(xs), **plot_keywords, linewidth=3, alpha=0.8)


def cumulative_plot(data, ax, max_, percentiles=[25, 50, 75], **plot_keywords):
    ecdf = sm.distributions.ECDF(data)
    x = np.array([0] + sorted(list(data)) + [max_])
    y = ecdf(x)
    plot = ax.step(x, y, **plot_keywords, linewidth=3, alpha=0.8)
    c = plot[0].get_c()
    l = plot[0].get_linestyle()
    for p in percentiles:
        ax.axvline(np.percentile(data, p), c=c, linestyle=l, alpha=0.8)
