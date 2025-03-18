import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_quartiles(series):
    percentage_rank = series.rank(method="max", pct=True)
    quartiles = np.empty_like(series, dtype="object")
    quartiles[percentage_rank < 0.25] = "Q1"
    quartiles[percentage_rank >= 0.25] = "Q2"
    quartiles[percentage_rank >= 0.50] = "Q3"
    quartiles[percentage_rank >= 0.75] = "Q4"
    return quartiles


def quartiles_confusion_plot(s1, s2, ax, splits=None, cbar=False, round_=None):
    if splits is None:
        splits = np.zeros(len(s1), dtype="int")
    m = np.zeros((4, 4))
    split_inds = np.unique(splits)
    for split_ind in split_inds:
        split_mask = splits == split_ind
        qs1 = get_quartiles(s1[split_mask])
        qs2 = get_quartiles(s2[split_mask])
        m = m + confusion_matrix(qs1, qs2, normalize="true")
    m = m / len(split_inds)
    if round_:
        m = m.round(round_)
    lab = ["Q1", "Q2", "Q3", "Q4"]
    sns.heatmap(
        m,
        annot=True,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cbar=cbar,
        xticklabels=lab,
        yticklabels=lab,
    )


def rank_rank_plot(s1, s2, ax):
    s1_idxs = np.argsort(s1)
    s2_idxs = np.argsort(s2)
    s1_ranks = np.arange(1, len(s1_idxs) + 1)[np.argsort(s1_idxs)]
    s2_ranks = np.arange(1, len(s2_idxs) + 1)[np.argsort(s2_idxs)]
    ax.scatter(s2_ranks, s1_ranks, s=1)
