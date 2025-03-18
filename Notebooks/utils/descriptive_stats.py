import os
from copy import deepcopy

import numpy as np
import pandas as pd
from rpy2 import robjects


def get_table(df, cols_num, cols_bin, name, show_count):
    df = deepcopy(df)
    cols = dict(**cols_num, **cols_bin)
    for c in cols_bin.keys():
        df[c] = df[c].apply(lambda v: 1 if v == "Yes" else (0 if v == "No" else v))
    df = df[list(cols.keys()) + ["sk1_id"]]
    df = get_descriptive_stats(df)
    df = df.rename(index=cols)
    df.loc[list(cols_bin.values()), "mean"] = (
        df.loc[list(cols_bin.values()), "mean"] * 100
    )
    for c in df.columns:
        if c == "count":
            df[c] = df[c].astype(int).astype(str)
        else:
            df[c] = df[c].round(decimals=1).astype(str)
    # minutes to hh:mm
    for r in df.index:
        if r[-7:] == "[HH:MM]":
            for c in df.columns:
                if c != "count":
                    df.loc[r, c] = (
                        f"{str(int(round(float(df.loc[r,c]), 0)) // 60).zfill(2)}:"
                        f"{str(int(round(float(df.loc[r,c]), 0)) % 60).zfill(2)}"
                    )
    df[name] = (
        df["mean"]
        + "("
        + df["mean_err"]
        + ")"
        + " +- "
        + df["std"]
        + "("
        + df["std_err"]
        + ")"
    )
    if show_count:
        df[name] = df[name] + " [N=" + df["count"] + "]"
    df_bin = df.loc[list(cols_bin.values())]
    df.loc[list(cols_bin.values()), name] = (
        df_bin["mean"] + "% [N=" + df_bin["count"] + "]"
    )
    return df[[name]]


def get_descriptive_table(df, cols_num, cols_bin, show_count=True):
    df_sk1 = df[df["source"] == "sk1"]
    sk1 = get_table(
        df_sk1, cols_num, cols_bin, f"Baseline [N={len(df_sk1)}]", show_count
    )
    df_sk2 = df[df["source"] == "sk2"]
    sk2 = get_table(
        df_sk2, cols_num, cols_bin, f"Follow-up [N={len(df_sk2)}]", show_count
    )
    all_ = get_table(df, cols_num, cols_bin, f"Pooled [N={len(df)}]", show_count)
    return sk1.join(sk2).join(all_)


def get_descriptive_stats(df):
    df_path = os.getcwd().replace("\\", "/") + "/temp.csv"
    df.reset_index().to_csv(df_path)
    base_str = (
        lambda col: f"""
        library(survey)
        df <- read.csv("{df_path}")
        df <- subset(df, !is.na({col}))
        df_design <- svydesign(data=df, ids=df$sk1_id)
    """
    )
    get_mean = lambda col: robjects.r(
        base_str(col) + "\n" + f"print(svymean(~ {col}, design=df_design))"
    )
    get_std = lambda col: robjects.r(
        base_str(col) + "\n" + f"print(svyvar(~ {col}, design=df_design))"
    )
    get_count = lambda col: robjects.r(base_str(col) + "\n" + f"print(nrow(df))")
    data = []
    cols = [col for col in df.columns if col != "sk1_id"]
    for col in cols:
        col_data = [
            *get_mean(col),
            *([v**0.5 for v in get_std(col)]),
            get_count(col)[0],
        ]
        if col_data[-1] == 0:
            col_data = [np.nan, np.nan, np.nan, np.nan, 0]
        data.append(col_data)
    res = pd.DataFrame(
        data, index=cols, columns=["mean", "mean_err", "std", "std_err", "count"]
    )
    os.remove(df_path)
    return res
