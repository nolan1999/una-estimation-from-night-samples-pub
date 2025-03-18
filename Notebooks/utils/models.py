import re

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import statsmodels.formula.api as sfapi
from rpy2.robjects import pandas2ri
from scipy import stats
from sklearn.preprocessing import StandardScaler

from bland_altman import corr_test


def mixedlm_backward_sel(
    data_train, data_test, id_col="sk1_id", target_col="target", sig=0.05
):
    while True:
        model = sfapi.mixedlm(
            f"{target_col} ~ {' + '.join(set(data_train.columns) - set([target_col, id_col]))}",
            data=data_train,
            groups=id_col,
        )
        model = model.fit(reml=False)  # to get non-nan aic, bic
        pvals = model.pvalues
        pvals = pvals[~(pvals.index == "Intercept")]
        if pvals.max() <= sig:
            break
        else:
            pred_rem = pvals.idxmax()
            data_train = data_train.loc[:, data_train.columns != pred_rem]
            data_test = data_test.loc[:, data_test.columns != pred_rem]
    predictions = model.predict(data_test)
    return model, predictions.to_numpy(), (model.aic, model.bic, model.params.to_dict())


def glmm_lasso(
    data_train,
    data_test,
    id_col="sk1_id",
    target_col="target",
    lambda_val=1.0,
    final_re=False,
    se=False,
):
    pandas2ri.activate()
    ro.r(
        """
    fit_glmmLasso <- function(train_df, test_df, id_col, target_col, lambda_val, final_re, se) {
        library(glmmLasso)
        set.seed(7)
        
        # Convert id_col to a factor (important for random effects)
        train_df[[id_col]] <- as.factor(train_df[[id_col]])
        test_df[[id_col]] <- as.factor(test_df[[id_col]])

        # Convert all predictors to numeric
        num_vars <- setdiff(names(train_df), c(target_col, id_col))
        for (var in num_vars) {
            train_df[[var]] <- as.numeric(train_df[[var]])
            test_df[[var]] <- as.numeric(test_df[[var]])
        }
        
        # Convert target to numeric
        train_df[[target_col]] <- as.numeric(train_df[[target_col]])
        
        # Build formula dynamically
        predictors <- setdiff(names(train_df), c(target_col, id_col))
        formula_str <- paste(target_col, "~", paste(predictors, collapse = " + "))
        formula <- as.formula(formula_str)
        rnd_effects <- list()
        rnd_effects[[id_col]] <- ~ 1  # Random intercept for id_col
        
        # Fit glmmLasso model (possibly with CV)
        if (is.vector(lambda_val) && length(lambda_val) > 1) {
       
            # This is buggy, i don't know why
            # (it is always selecting 0, even when other lambdas are performing better...)
            # so i reimplement the CV myself instead
            #library(cv.glmmLasso)
            #cv_result <- cv.glmmLasso(
            #    formula, rnd = rnd_effects, data = train_df, lambdas = lambda_val,
            #    lambda.final = "lambda.1se", final.re = final_re
            #)
            #lambda_sel <- cv_result$lambda.1se
            #model <- cv_result$glmmLasso.final

            # inspired from https://github.com/cran/glmmLasso/blob/master/demo/glmmLasso-soccer.r
            N <- dim(train_df)[1]
            ind <- sample(N, N)
            kk <- 5
            nk <- floor(N/kk)
            lambda_errors <- matrix(NA, nrow=length(lambda_val), ncol=kk)
            for(j in 1:length(lambda_val)) {
                lambda_error <- 0
                for (i in 1:kk) {
                    if (i < kk) {
                        indi <- ind[((i-1)*nk+1): (i*nk)]
                    } else {
                        indi <- ind[((i-1)*nk+1):N]
                    }
                    fold_train <- train_df[setdiff(1:N, indi),]
                    fold_test <- train_df[indi,]
                    tryCatch({
                        fold_model <- glmmLasso(
                            formula, rnd = rnd_effects, data = fold_train, lambda = lambda_val[j]
                        )
                        fold_pred <- predict(fold_model, fold_test)
                        lambda_errors[j, i] <- mean((fold_pred - fold_test[[target_col]])^2)
                    }, error = function(e) {
                        lambda_errors[j, i] <- Inf
                    })
                }
            }
            mean_errors <- rowMeans(lambda_errors, na.rm = TRUE)
            print(mean_errors)
            best_idx <- which.min(mean_errors)
            if(se) {
                best_mse <- mean_errors[best_idx]
                best_std_error <- sd(lambda_errors[best_idx, ], na.rm = TRUE) / sqrt(kk)
                lambda_sel <- max(lambda_val[mean_errors <= (best_mse + best_std_error)])
            } else {
                lambda_sel <- lambda_val[best_idx]
            }
            model <- glmmLasso(
                formula, rnd = rnd_effects, data = train_df, lambda = lambda_sel, final.re = final_re
            )
        } else {
            lambda_sel <- lambda_val
            model <- glmmLasso(
                formula, rnd = rnd_effects, data = train_df, lambda = lambda_sel, final.re = final_re
            )
        }
        features <- names(model$coefficients)
        coefficients <- as.numeric(model$coefficients)
        predictions <- predict(model, test_df)
        return(list(
            model = model, lambda_val = lambda_sel, predictions = predictions,
            features = features, coefficients = coefficients
        ))
    }
    """
    )

    data_train[id_col] = data_train[id_col].astype("category")
    data_test[id_col] = data_test[id_col].astype("category")
    data_test[target_col] = 0.0  # somehow, this is required
    lambda_val_r = (
        ro.FloatVector(lambda_val) if isinstance(lambda_val, list) else lambda_val
    )

    # Fit model
    r_data_train = pandas2ri.py2rpy(data_train)
    r_data_test = pandas2ri.py2rpy(data_test)
    r_func = ro.r["fit_glmmLasso"]
    result = r_func(
        r_data_train, r_data_test, id_col, target_col, lambda_val_r, final_re, se
    )

    # Extract model and predictions
    predictions = np.array(result.rx2("predictions")).flatten()
    model = result.rx2("model")
    lambda_sel = result.rx2("lambda_val")
    print(f"Selected lambda: {lambda_sel}")
    if isinstance(lambda_val, list) and (
        lambda_sel
        in (min(lambda_val), min(l for l in lambda_val if l != 0), max(lambda_val))
    ):
        print(
            f"WARNING: Selected lambda ({lambda_sel}) is at the edge of the search range. "
            "Consider expanding the search range."
        )
    aic = model.rx2("aic")[0][0]
    bic = model.rx2("bic")[0][0]
    coefficients = dict(zip(result.rx2("features"), result.rx2("coefficients")))

    return model, predictions, (aic, bic, coefficients)


# def lmm_seagull(data_train, data_test, id_col='sk1_id', target_col='target', alpha=1.0):
#     ro.r('''
#     fit_seagull <- function(train_df, test_df, id_col, target_col, alpha) {
#         library(seagull)

#         # Convert id_col to a factor (important for random effects)
#         train_df[[id_col]] <- as.factor(train_df[[id_col]])
#         test_df[[id_col]] <- as.factor(test_df[[id_col]])

#         # Convert all predictors to numeric
#         num_vars <- setdiff(names(train_df), c(target_col, id_col))
#         for (var in num_vars) {
#             train_df[[var]] <- as.numeric(train_df[[var]])
#             test_df[[var]] <- as.numeric(test_df[[var]])
#         }

#         # Convert target to numeric
#         train_df[[target_col]] <- as.numeric(train_df[[target_col]])

#         # Build formula dynamically
#         predictors <- setdiff(names(train_df), c(target_col, id_col))
#         formula_str <- paste(target_col, "~", paste(predictors, collapse = " + "))
#         formula <- as.formula(formula_str)

#         # Prepare design matrices
#         X <- model.matrix(formula, data = train_df)[, -1]  # Exclude intercept
#         Z <- model.matrix(~ 0 + train_df[[id_col]])  # Random effects design matrix

#         # Fit Lasso-regularized mixed model using seagull
#         model <- seagull(
#             y = train_df[[target_col]],
#             X = X,
#             Z = Z,
#             alpha = alpha,
#             standardize = FALSE
#         )

#         # Get the coefficients (fixed effects)
#         coefficients <- model$fixed_effects

#         # Make predictions on test set
#         X_test <- model.matrix(formula, data = test_df)[, -1]
#         Z_test <- model.matrix(~ 0 + test_df[[id_col]])
#         predictions <- predict(model, newdata = data.frame(X = X_test, Z = Z_test))

#         return(list(model = model, predictions = predictions, coefficients = coefficients))
#     }
#     ''')

#     # Convert pandas DataFrames to R DataFrames
#     r_data_train = pandas2ri.py2rpy(data_train)
#     r_data_test = pandas2ri.py2rpy(data_test)

#     # Call the R function
#     r_func = ro.r['fit_seagull']
#     result = r_func(r_data_train, r_data_test, id_col, target_col, alpha)

#     # Extract model, predictions, and coefficients
#     predictions = np.array(result.rx2('predictions'))  # Predictions on test set
#     model = result.rx2('model')  # The trained seagull model
#     coefficients = dict(zip(result.rx2('coefficients').names, result.rx2('coefficients')))

#     return model, predictions, (0., 0., coefficients)


def get_metrics(data, preds, true):
    # significant difference bias hypertension
    def mean(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    err = preds - true
    sig_diff = stats.permutation_test(
        (
            err[data["hypertension_combined"] == "No"],
            err[data["hypertension_combined"] == "Yes"],
        ),
        mean,
        vectorized=True,
        n_resamples=1e4,
        permutation_type="independent",
        alternative="two-sided",
        random_state=42,
    ).pvalue
    # significant trend
    sig_trend = corr_test(true, err)
    # RMSE
    rmse = (((preds - true) ** 2).mean()) ** 0.5
    corr = stats.pearsonr(true, err)
    pval_corr = corr.pvalue
    corr = corr.statistic
    r2 = (np.corrcoef(true, preds)[0][1]) ** 2
    return sig_diff, sig_trend, corr, pval_corr, rmse, r2


def get_predictors(coefficients, kernels):
    predictors = []
    used_predictors = []
    features = []
    used_features = []
    for predictor, coefficient in coefficients.items():
        feature = predictor
        if "Intercept" in predictor or "sk1_id" in predictor:
            continue
        if "_INT_" in predictor:
            feature = feature.split("_INT_")[0]
        feature = feature.split("_SUB")[0]
        if kernels:
            for kernel in kernels:
                feature = feature.split(f"_{kernel}")[0]
        predictors.append(predictor)
        features.append(feature)
        if coefficient != 0.0:
            used_predictors.append(predictor)
            used_features.append(feature)
    predictors = list(set(predictors))
    used_predictors = list(set(used_predictors))
    features = list(set(features))
    used_features = list(set(used_features))
    return {
        "predictors": predictors,
        "used_predictors": used_predictors,
        "features": features,
        "used_features": used_features,
    }


def preprocess(
    X_train, X_test, categorical_cols, numerical_cols, labels_df, kernels, inter_col
):
    # One-hot encoding of categorical variables
    # (one category less [drop_first=True] to keep full matrix rank)
    for col in categorical_cols:
        X_train = pd.concat(
            (
                X_train,
                pd.get_dummies(X_train[col], prefix=f"{col}_SUB", drop_first=True),
            ),
            axis=1,
        ).drop(col, axis=1)
        X_test = pd.concat(
            (X_test, pd.get_dummies(X_test[col], prefix=f"{col}_SUB", drop_first=True)),
            axis=1,
        ).drop(col, axis=1)
    categorical_cols = [col for col in X_train.columns if col[-4:] == "_SUB"]

    new_numerical_cols = numerical_cols.copy()
    # Interaction column
    if inter_col:
        for col in numerical_cols + categorical_cols:
            if col != inter_col and not (
                col.split("_")[-1] == "min"
                and inter_col.split("_")[-1] == "mmolh"  # ensure invertible
            ):
                prod_col = f"{col}_INT_{inter_col}"
                X_train[prod_col] = X_train[inter_col] * X_train[col]
                X_test[prod_col] = X_test[inter_col] * X_test[col]
                new_numerical_cols.append(prod_col)
                labels_df.loc[prod_col] = (
                    f"{labels_df.loc[col]}*{labels_df.loc[inter_col]}"
                )
    # Kernels
    if kernels:
        for k_name, k_fun in kernels.items():
            X_train_kernel = (
                X_train[numerical_cols]
                .apply(k_fun)
                .rename(columns=lambda col: f"{col}_{k_name}")
            )
            X_train = pd.concat([X_train, X_train_kernel], axis=1)
            X_test_kernel = (
                X_test[numerical_cols]
                .apply(k_fun)
                .rename(columns=lambda col: f"{col}_{k_name}")
            )
            X_test = pd.concat([X_test, X_test_kernel], axis=1)
            new_numerical_cols.extend(list(X_train_kernel.columns))
            for col in numerical_cols:
                labels_df.loc[col + "_" + k_name] = labels_df.loc[col] + " " + k_name

    # Standardize
    feature_cols = new_numerical_cols + categorical_cols
    scaler = StandardScaler()
    X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])
    return X_train, X_test


def test_model(
    data,
    cols,
    target_col,
    cat_cols,
    labels_df,
    kernels=None,
    inter_col=None,
    norm_out=True,
    model_fn=glmm_lasso,
    **model_kwargs,
):
    # TODO: add 'center' to models? 'source'?
    nonfeat_cols = ["sk1_id", "cv_split"]
    X = data.loc[:, cols + nonfeat_cols].copy()
    y = data[target_col]
    categorical_cols = set(X.columns) & set(cat_cols) - set(nonfeat_cols)
    print(f"{len(categorical_cols)} categorical columns. Will encode.")
    numerical_cols = set(X.columns) - set(cat_cols) - set(nonfeat_cols)
    print(f"{len(numerical_cols)} numerical columns. Will standardize.")

    # result stores
    cv_preds = pd.Series(index=X.index, data=np.nan, name="preds")
    cv_results = [[], [], [], [], [], []]
    aics = []
    bics = []
    predictors = []

    for cv_split in sorted(data["cv_split"].unique()):
        subset_id = data[data["cv_split"] == cv_split].index
        X_test = X[X.index.isin(subset_id)].copy().drop(columns=["cv_split"])
        y_test = y[y.index.isin(subset_id)].copy()
        X_train = X[~(X.index.isin(subset_id))].copy().drop(columns=["cv_split"])
        y_train = y[~(y.index.isin(subset_id))].copy()
        # Preprocess
        X_train, X_test = preprocess(
            X_train,
            X_test,
            list(categorical_cols),
            list(numerical_cols),
            labels_df,
            kernels,
            inter_col,
        )
        # Prepare input data
        data_train = X_train.copy()
        if norm_out:
            scaler_y = StandardScaler()
            data_train["target"] = scaler_y.fit_transform(
                y_train.to_numpy().reshape(-1, 1)
            ).flatten()
        else:
            data_train["target"] = y_train
        # Get model and predictions
        _, preds, (aic, bic, coefficients) = model_fn(
            data_train.copy(), X_test.copy(), **model_kwargs
        )
        if norm_out:
            preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
        for r, l in zip(
            get_metrics(data[data["cv_split"] == cv_split], preds, y_test), cv_results
        ):
            l.append(r)
        cv_preds[cv_preds.index.isin(subset_id)] = preds
        aics.append(aic)
        bics.append(bic)
        predictors.append(get_predictors(coefficients, kernels))
    predictors_merged = {
        k: [predictor[k] for predictor in predictors] for k in predictors[0]
    }
    cv_metrics = tuple(np.mean(l) for l in cv_results) + (
        np.mean(aics),
        np.mean(bics),
        predictors_merged,
    )
    # pooled_metrics = get_metrics(data, cv_preds, y)

    # Train on whole data (no CV)
    X_train = X.copy().drop(columns=["cv_split"])
    X_train, _ = preprocess(
        X_train,
        X_train.copy(),
        list(categorical_cols),
        list(numerical_cols),
        labels_df,
        kernels,
        inter_col,
    )
    data_train = X_train.copy()
    if norm_out:
        scaler_yall = StandardScaler()
        data_train["target"] = scaler_yall.fit_transform(
            y.to_numpy().reshape(-1, 1)
        ).flatten()
    else:
        data_train["target"] = y
    _, preds, (aic, bic, coefficients) = model_fn(data_train, X_train, **model_kwargs)
    if norm_out:
        preds = scaler_yall.inverse_transform(preds.reshape(-1, 1)).flatten()
    metrics = get_metrics(data, preds, y) + (
        aic,
        bic,
        get_predictors(coefficients, kernels),
    )
    print(coefficients)

    return (
        (preds, cv_preds),  # preds
        (metrics[0], cv_metrics[0]),  # sig diff
        (metrics[1], cv_metrics[1]),  # sig trend
        (metrics[2], cv_metrics[2]),  # corr
        (metrics[3], cv_metrics[3]),  # pval corr
        (metrics[4], cv_metrics[4]),  # rmse
        (metrics[5], cv_metrics[5]),  # r2
        (metrics[6], cv_metrics[6]),  # aic
        (metrics[7], cv_metrics[7]),  # bic
        (metrics[8], cv_metrics[8]),  # predictors
    )


def get_models_df_preds(
    models,
    data,
    target_col,
    cat_cols,
    labels_df,
    kernels=None,
    inter_col=None,
    norm_out=False,
    model_fn=glmm_lasso,
    **model_kwargs,
):
    names = []
    npredictors = []
    predictors = []
    r2s = []
    aics = []
    bics = []
    rmses = []
    sig_diffs = []
    sig_trends = []
    corrs = []
    preds = []

    for label, cols in models.items():
        print(f"\nModel {label}:")
        preds_, sig_diff, sig_trend, corr, pval_corr, rmse, r2, aic, bic, predictor = (
            test_model(
                data,
                cols,
                target_col,
                cat_cols,
                labels_df,
                kernels=kernels,
                inter_col=inter_col,
                norm_out=norm_out,
                model_fn=model_fn,
                **model_kwargs,
            )
        )
        print(f"\nRMSE {rmse}")
        names.append(label)
        len_p = lambda p: (
            np.mean([len(p_) for p_ in p]) if isinstance(p[0], list) else len(p)
        )
        npredictor_descr = lambda p: (
            f'{len_p(p["used_predictors"])}/{len_p(p["predictors"])} | '
            f'{len_p(p["used_features"])}/{len_p(p["features"])}'
        )
        npredictors.append(
            f"{npredictor_descr(predictor[0])} ({npredictor_descr(predictor[1])})"
        )
        r2s.append(f"{round(r2[0], 3)} ({round(r2[1], 3)})")
        aics.append(f"{round(aic[0])} ({round(aic[1])})")
        bics.append(f"{round(bic[0])} ({round(bic[1])})")
        rmses.append(f"{round(rmse[0], 3)} ({round(rmse[1], 3)})")
        sig_diffs.append(f"{round(sig_diff[0], 3)} ({round(sig_diff[1], 3)})")
        sig_trends.append(f"{round(sig_trend[0], 3)} ({round(sig_trend[1], 3)})")
        corrs.append(
            f"{round(corr[0], 3)} | {round(pval_corr[0], 3)} ({round(corr[1], 3)} | {round(pval_corr[1], 3)})"
        )

        def get_pred_name(pred_lab):
            if "Intercept" in pred_lab:
                return "Intercept"
            if "_INT_" in pred_lab:
                pred_lab_1, pred_lab_2 = pred_lab.split("_INT_")
                return f"{get_pred_name(pred_lab_1)} x {get_pred_name(pred_lab_2)}"
            var_name = labels_df.loc[
                re.sub("_SUB_.*", "", pred_lab).replace(" Var", ""), "Variable Label"
            ]
            pred_name = (
                var_name + ": " + re.sub(".*_SUB_", "", pred_lab)
                if "_SUB_" in pred_lab
                else var_name
            )
            return pred_name

        predictors.append(
            sorted([get_pred_name(c) for c in predictor[0]["used_predictors"]])
        )
        preds.append(preds_)

    models_df = pd.DataFrame(
        {
            "Model": names,
            "Number of predictors": npredictors,
            "R2": r2s,
            "AIC": aics,
            "BIC": bics,
            "Root-mean-square error": rmses,
            "P-value difference hypertensive-normotensive": sig_diffs,
            "P-value trend measured-error": sig_trends,
            "Correlation measured-error": corrs,
            "Predictors": predictors,
        }
    )

    return models_df, preds
