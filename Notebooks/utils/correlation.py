import numpy as np
import statsmodels.api as sm
from mlinsights.mlmodel import IntervalRegressor
from sklearn.linear_model import LinearRegression


def plot_correlation(x, y, ax, fit_intercept=True):
    ax.scatter(x, y, alpha=0.3, s=5)
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(x.values.reshape(-1, 1), y)
    print("Coeffs:", LR.coef_)
    r_squared = LR.score(x.values.reshape(-1, 1), y)
    ax.plot(sorted(x), LR.predict(np.array(sorted(x.values.reshape(-1, 1)))), color="k")
    res = sm.OLS(y, sm.add_constant(x)).fit()
    print(res.t_test([0, 1]))
    return r_squared


def plot_regression(x, y, ax, fit_intercept=True, show_id=True):
    n_estimators = 1000
    r_squared = plot_correlation(x, y, ax, fit_intercept=fit_intercept)
    lin = IntervalRegressor(
        LinearRegression(fit_intercept=fit_intercept), n_estimators=n_estimators
    )
    lin.fit(x.values.reshape(-1, 1), y.values)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_test = np.linspace(*xlim, 1000).reshape(-1, 1)
    bootstrapped_pred = lin.predict_sorted(x_test)
    lower = bootstrapped_pred[:, int(5 * n_estimators / 100) - 1]
    upper = bootstrapped_pred[:, int(95 * n_estimators / 100) - 1]
    ax.fill_between(x_test.flatten(), lower, upper, color="k", alpha=0.2)
    if show_id:
        ax.plot(x_test, x_test, c="k", linestyle="--")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return r_squared


def plot_group_differences(x, y, groups, ax, fit_intercept=True, show_id=False):
    r_squared = dict()
    x_test = np.linspace(0, x.max(), 10)
    for g, m in zip(set(groups[~groups.isna()].values), ["x", "d"]):
        x_ = x[groups == g].values
        y_ = y[groups == g].values
        ax.scatter(x_, y_, alpha=0.2, s=5, marker=m)
        LR = LinearRegression(fit_intercept=fit_intercept).fit(x_.reshape(-1, 1), y_)
        r_squared_ = LR.score(x_.reshape(-1, 1), y_)
        r_squared[g] = r_squared_
        ax.plot(
            x_test,
            LR.predict(x_test.reshape(-1, 1)),
            label=f"{g}: $R^2$={round(r_squared_, 3)}",
            marker=m,
        )
    if show_id:
        ax.plot(x_test, x_test, c="k", linestyle="--")
    ax.legend()
    ax.set_xlim(left=0, right=x.max())
    ax.set_ylim(bottom=0)
    return r_squared
