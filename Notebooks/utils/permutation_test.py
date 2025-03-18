import numpy as np
from scipy.stats import permutation_test


def permutation_test_repeated(
    x,
    y,
    x_id,
    y_id,
    statistic,
    *,
    permutation_type="independent",
    n_resamples=9999,
    alternative="two-sided",
    random_state=None,
):
    if permutation_type == "pairings":
        # nothing to do here (observations remain in the same sample)
        return permutation_test(
            (x, y),
            statistic,
            permutation_type=permutation_type,
            n_resamples=n_resamples,
            alternative=alternative,
            random_state=random_state,
        )
    x_id = np.array([f"x_{x_id_}" for x_id_ in x_id])
    y_id = np.array([f"y_{y_id_}" for y_id_ in y_id])
    x_argsort = np.argsort(x_id)
    x_sorted = np.array(x)[x_argsort]
    x_id_sorted = x_id[x_argsort]
    y_argsort = np.argsort(y_id)
    y_sorted = np.array(y)[y_argsort]
    y_id_sorted = y_id[y_argsort]

    samples_x = np.sort(np.unique(x_id_sorted))
    samples_y = np.sort(np.unique(y_id_sorted))

    def statistic_tf(x_ids, y_ids, axis=None):
        mask_xx = np.isin(x_id_sorted, x_ids)
        mask_xy = np.isin(y_id_sorted, x_ids)
        mask_yy = np.isin(y_id_sorted, y_ids)
        mask_yx = np.isin(x_id_sorted, y_ids)
        return statistic(
            np.concatenate([x_sorted[mask_xx], y_sorted[mask_xy]]),
            np.concatenate([y_sorted[mask_yy], x_sorted[mask_yx]]),
            axis=axis,
        )

    return permutation_test(
        (samples_x, samples_y),
        statistic_tf,
        permutation_type=permutation_type,
        n_resamples=n_resamples,
        alternative=alternative,
        random_state=random_state,
    )
