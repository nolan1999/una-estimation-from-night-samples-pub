import json
import os

import numpy as np
import pandas as pd
from mlinsights.mlmodel import IntervalRegressor
from rpy2 import robjects
from scipy.stats import (
    bootstrap,
    shapiro,
    normaltest,
    anderson,
    pearsonr,
    permutation_test,
)
from sklearn.linear_model import LinearRegression


def corr(x, y):
    return pearsonr(x, y).statistic


def corr_test(x, y):
    return permutation_test(
        (x, y),
        corr,
        vectorized=False,
        permutation_type="pairings",
        alternative="two-sided",
        n_resamples=1e4,
        random_state=42,
    ).pvalue


def plot_regression(x, y, ax):
    n_estimators = 5000
    lin = IntervalRegressor(
        LinearRegression(fit_intercept=True), n_estimators=n_estimators
    )
    lin.fit(x.values.reshape(-1, 1), y)
    x_test = np.linspace(*ax.get_xlim(), 1000).reshape(-1, 1)
    bootstrapped_pred = lin.predict_sorted(x_test)
    lower = bootstrapped_pred[:, int(1 * n_estimators / 100) - 1]
    upper = bootstrapped_pred[:, int(99 * n_estimators / 100) - 1]
    ax.fill_between(x_test.flatten(), lower, upper, color="y", alpha=0.2)


# 1:1 from https://github.com/johnjdavisiv/rmba/blob/master/rmba.R
RMBA = """
#Repeated measures bland altman with mixed models
#Following Parker et al 2016 (doi:10.1371/journal.pone.0168321)
#John J Davis IV
#17 May 2020

#USAGE EXAMAPLE: 
#rmba_res <- rmba(my_df, "RRox", "RRacc", "Activity", "PatientID")

#Input: 
# data - dataframe
# measure_one_col - string name of column with first measure (or gold standard)
# measure_two_col - string name of column with second measure
# condition_col - name of column with (factor) conditions. Leave blank to not adjust for any conditions
# id_col - string name of column with IDs (patients, etc.)
# loa_level - limit of agreement percent (default: 0.95 for 95% LoA)
# verbose - print results?
# bootstrap - use parametric bootstrap to estimate CIs? (Experimental!)
# bootstrap_ci_level - confidence level for bootstrapping. Defaults to 95%
# B - number of bootstrap resamples 
# seed - rng seed for bootstrapping

#Output:
# rmba_results: a list with the mean bias, se of mean bias, total SD, and limits of agreement
# Results are returned as measure two COMPARED TO measure one. 
# If one of your measures is a gold standard, it should be measure one. 


#-------------------------------------------------------
#         Primary function   
#-------------------------------------------------------

rmba <- function(data, 
                 measure_one_col, 
                 measure_two_col, 
                 condition_col = "1", 
                 id_col,
                 loa_level = 0.95,
                 verbose = TRUE,
                 bootstrap = FALSE,
                 bootstrap_ci_level = 0.95,
                 B = 10000,
                 seed = NA) {

  require(nlme)
  
  #In case user wants something other than 95% LoA
  se_mult <- qnorm(1-(1-loa_level)/2)
  
  #Unlist for tibble messiness
  data$measure_diff_rmba <- unlist(data[,measure_two_col] - data[,measure_one_col])
  
  #Doing measure two minus one, so if measure one is gold standard, a positive result means
  #that measure two is an OVERESTIMATE (+)
  
  lme_form <- as.formula(paste("measure_diff_rmba ~ ", condition_col, sep=""))
  lme_id_form <- as.formula(paste("~1|", id_col, sep=""))
  
  model_one <- lme(lme_form, random = lme_id_form,
                   correlation = corCompSymm(form = lme_id_form),
                   data = data,
                   na.action = na.omit)
  
  #Within-subject SD is the residual SD
  within_sd <- as.numeric(VarCorr(model_one)[2,2])
  #Between subject SD is the random intercept SD
  between_sd <- as.numeric(VarCorr(model_one)[1,2])
  
  #Total SD is the +/- for the mean bias (adjusted for condition)
  total_sd <- sqrt(between_sd^2 + within_sd^2)
  
  #Model two: intercept only
  #"extracts appropriately weighted mean and standard error" - Parker et al.
  model_two <- lme(measure_diff_rmba ~ 1, random = lme_id_form,
                   correlation = corCompSymm(form = lme_id_form),
                   data = data,
                   na.action = na.omit)
  
  #Intercept of the intercept only model is the mean bias
  #  The standard error of that metric is the SE of the mean bias
  mean_bias <- summary(model_two)$tTable[1,1]
  mean_bias_se <- summary(model_two)$tTable[1,2]
  
  #Calculate 95% (or whatever percent) limits of agreement
  lo_limit <- mean_bias - se_mult*total_sd
  hi_limit <- mean_bias + se_mult*total_sd
  
  #If bootstrap estimates desired, run boot
  if (bootstrap){
    #Do bootstrap
    
    if (!is.na(seed)) set.seed(seed)
    
    boot_results <- matrix(nrow=B, ncol=5)
    colnames(boot_results) <- c("bias","bias_se","sd",
                            "lower_agreement_limit",
                            "upper_agreement_limit")
    print("CAUTION! Parametric bootstrap implementation differs from Parker et al. This feature should be considered experimental.")
    print("Running bootstrap; this could take a while...")
    
    prog_bar <- txtProgressBar(min = 0, max = B, initial = 0, style=1) 
    
    #For each resample...
    for (b in 1:B){
      #Get a bootstrap sample
      boot_results[b,] <- unlist(rmba_resample(data, model_one, 
                                               condition_col,id_col, loa_level))
      setTxtProgressBar(prog_bar,b)
    }
    
    #Probabilities for bootstrap confidence level
    boot_probs <- c((1-bootstrap_ci_level)/2, 
                    1-(1-bootstrap_ci_level)/2)
    
    #Get percentiles at end
    boot_ci <- apply(boot_results, 2, quantile, probs= boot_probs)
    
  } else boot_ci <- NULL #if no bootstrap requested
  
  #Write results to list
  rmba_results <- list(bias = mean_bias,
                       bias_se = mean_bias_se,
                       sd = total_sd,
                       lower_agreement_limit = lo_limit,
                       upper_agreement_limit = hi_limit,
                       boot_ci = boot_ci)
  
  #Print results if desired
  if (verbose) {
    print(sprintf("Mean bias of %s compared to %s: %.3f with %.0f%% limits of agreement [%.3f, %.3f]",
                  measure_two_col,
                  measure_one_col,
                  rmba_results$bias,
                  loa_level*100,
                  rmba_results$lower_agreement_limit,
                  rmba_results$upper_agreement_limit))
  }
  
  return(rmba_results)

}
  

#-------------------------------------------------------
#   Parametric boostrapping for confidence intervals
#-------------------------------------------------------

#This function does the resampling. See below for function that fits the resampled data
rmba_resample <- function(data, 
                          orig_model_one, 
                          condition_col,
                          id_col,
                          loa_level) {
  #Input values needed:
  # orig_model_one - original model fit in call to rmba()
  # loa_level - original loa_level
  
  #Perform parametric bootstrap 
  #Specifically the "parametric random effects bootstrap coupled with residual bootstrap"
  #in section 2.3.2 in Thai et al 2013. Pharm Stat 12(3);129-140
  #The idea is we just take new Gaussian draws using the SD of random intercept and
  #the SD of the residuals to get a new Yij for each bootstrap replicate.
  
  #Grab SDs
  within_sd <- as.numeric(VarCorr(orig_model_one)[2,2])
  between_sd <- as.numeric(VarCorr(orig_model_one)[1,2])
  
  #How many subjects/clusters? 
  n_id_levels <- length(levels(as.factor(unlist(data[,id_col]))))
  
  #This gets a number of non-NA values for each patient
  n_i <- by(data$measure_diff_rmba, INDICES = unlist(data[,id_col]),
            FUN = function (x) sum(!is.na(x)))
  #Same as:
  #data %>%
  #  group_by(PatientID) %>%
  #  drop_na(measure_diff_rmba) %>%
  #  count()
  
  #Fixed-effects only (missing values will be omitted silently)
  X_B <- predict(orig_model_one, level=0)
  
  #Resample new random effects for each subject (n_i times) using estimated SD
  new_re <- rep(rnorm(n_id_levels, 0 , between_sd), 
                times = as.numeric(n_i))
  
  #Resample new residuals 
  #(all subjects share the same residual SD so we don't need to condition on subject here)
  new_resid <- rnorm(length(X_B),0,
                     within_sd)
  
  # Get new difference vector 
  #This is Yij = XB + Zu + e because Z is just a column vector of ones.
  boot_diff <- X_B + new_re + new_resid
  
  # --- Prepare data frame for boot_rmba()
  
  #NA ind - need to trim data because we used na.omit earlier
  na_ind <- is.na(data$measure_diff_rmba)
  
  #Get stuff ready for boot dataframe
  boot_id <- unlist(data[!na_ind, id_col])
  
  #If condition column was specificed
  if (condition_col != "1"){
    boot_condition <- unlist(data[!na_ind, condition_col])
  } else {
    boot_condition <- rep("1", length(boot_id))
  }
  
  #Make boot dataframe
  boot_data <- data.frame(boot_id = boot_id, 
                          boot_condition = boot_condition,
                          boot_diff = boot_diff)
  
  #Call boot_rmba and get estimates
  return(rmba_boot(boot_data, loa_level))
  
}

#-------------------------------------------------------
#         Fitting new model to resampled data     
#-------------------------------------------------------

#Helper function that refits a new model to resampled Yij difference values
rmba_boot <- function(boot_data,
                      loa_level = 0.95) {
  
  require(nlme)
  
  se_mult <- qnorm(1-(1-loa_level)/2)

  #If using a fixed effect for condition
  if (length(unique(boot_data$boot_condition)) == 1 &
    unique(boot_data$boot_condition)[1] == "1") {
    
    lme_form <- as.formula("boot_diff ~ 1")
    
  } else{
    lme_form <- as.formula("boot_diff ~ boot_condition")
  }
  
  lme_id_form <- as.formula("~1|boot_id")
  
  boot_model_one <- lme(lme_form, random = lme_id_form,
                   correlation = corCompSymm(form = lme_id_form),
                   data = boot_data,
                   na.action = na.omit)
  
  #Within-subject SD is the residual SD
  within_sd <- as.numeric(VarCorr(boot_model_one)[2,2])
  #Between subject SD is the random intercept SD
  between_sd <- as.numeric(VarCorr(boot_model_one)[1,2])
  
  #Total SD is the +/- for the mean bias (adjusted for condition)
  total_sd <- sqrt(between_sd^2 + within_sd^2)
  
  #Model two: intercept only
  #"extracts appropriately weighted mean and standard error
  boot_model_two <- lme(boot_diff ~ 1, random = lme_id_form,
                   correlation = corCompSymm(form = lme_id_form),
                   data = boot_data,
                   na.action = na.omit)
  
  #Intercept of the intercept only is the mean bias
  #  The standard error of that metric is the SE of the mean bias
  mean_bias <- summary(boot_model_two)$tTable[1,1]
  mean_bias_se <- summary(boot_model_two)$tTable[1,2]
  
  #Calculate 95% limits of agreement
  lo_limit <- mean_bias - se_mult*total_sd
  hi_limit <- mean_bias + se_mult*total_sd
  
  boot_rmba_results <- list(bias = mean_bias,
                       bias_se = mean_bias_se,
                       sd = total_sd,
                       lower_agreement_limit = lo_limit,
                       upper_agreement_limit = hi_limit)
  
  #use unlist() later to turn this into a named numeric
  return(boot_rmba_results)
}
"""


def get_ba_stats(x, y, id_col):
    current_dir = os.getcwd().replace("\\", "/")
    pd.DataFrame({"x": x, "y": y, "id_col": id_col}).to_csv(current_dir + "/temp.csv")
    robjects.r(
        f"""
        library(rjson)
        df <- read.csv("{current_dir + "/temp.csv"}")
        {RMBA}
        stats <- rmba(
            data = df,
            measure_one_col = "y",
            measure_two_col = "x",
            id_col = "id_col",
            bootstrap = TRUE,
            verbose = FALSE,
            B=500,
            seed = 42
        )
        jsonStats <- toJSON(stats)
        write(
          jsonStats,
          "{current_dir + "/temp.json"}")
    """
    )
    with open(current_dir + "/temp.json", "r") as f:
        stats = json.load(f)
    os.remove(current_dir + "/temp.csv")
    os.remove(current_dir + "/temp.json")
    return stats


def bland_altman_plot(
    data1,
    data2,
    dataid,
    ax,
    percent=False,
    relative=False,
    x_true=False,
    text_left=False,
    spacing=0.8,
    two_lines=True,
):
    mean = 0.5 * data1 + 0.5 * data2
    if x_true:
        xs = data1
        ax.set_xlabel("Measured [mmol]")
    else:
        xs = mean
        ax.set_xlabel("Mean (measured, predicted) [mmol]")
    if percent:
        data1 = data1 / mean
        data2 = data2 / mean
    elif relative:
        data2 = data2 / data1
        data1 = data1 / data1

    # have bias in the right direction
    ba_stats = get_ba_stats(data2, data1, dataid)
    diff = (data2 - data1).values
    # bias
    bias = ba_stats["bias"]
    # LOA
    lowerLOA = ba_stats["lower_agreement_limit"]
    upperLOA = ba_stats["upper_agreement_limit"]
    # CIs
    biasLowerCI, biasUpperCI = ba_stats["boot_ci"][:2]
    lowerLOA_lowerCI, lowerLOA_upperCI = ba_stats["boot_ci"][-4:-2]
    upperLOA_lowerCI, upperLOA_upperCI = ba_stats["boot_ci"][-2:]

    ax.scatter(xs, diff, s=1, alpha=0.2, c="k")
    # bias
    ax.axhline(0, c="k", linewidth=1)
    ax.axhline(bias, c="b", linewidth=1)
    xlim = ax.get_xlim()
    xlim = (0.0, xlim[1])
    ax.fill_between(
        [xlim[0], xlim[1] * 1.5],
        [biasUpperCI] * 2,
        [biasLowerCI] * 2,
        color="b",
        alpha=0.2,
    )
    # LOA
    ax.axhline(upperLOA, c="r", linewidth=1)
    ax.axhline(lowerLOA, c="r", linewidth=1)
    # xlim hack for multiple plots...
    ax.fill_between(
        [xlim[0], xlim[1] * 1.5],
        [upperLOA_upperCI] * 2,
        [upperLOA_lowerCI] * 2,
        color="r",
        alpha=0.2,
    )
    ax.fill_between(
        [xlim[0], xlim[1] * 1.5],
        [lowerLOA_upperCI] * 2,
        [lowerLOA_lowerCI] * 2,
        color="r",
        alpha=0.2,
    )
    plot_regression(xs, diff, ax)
    if percent:
        ax.set_ylabel("Mean-normalized difference")
    elif relative:
        ax.set_ylabel("Relative difference \n((predicted-measured)/measured)")
    else:
        ax.set_ylabel("Absolute difference \n(predicted-measured) [mmol]")
    xText = 0.2 if text_left else xlim[1] * spacing
    newline = "\n"
    ax.text(
        xText,
        lowerLOA + ((0.3 * (upperLOA - bias)) if not two_lines else 0),
        f'-1.96SD: {"%.1f" % lowerLOA} {newline if two_lines else ""} [{"%.1f" % lowerLOA_lowerCI}, {"%.1f" % lowerLOA_upperCI}]',
        ha="center",
        va="center",
    )
    ax.text(
        xText,
        bias + ((0.4 * (upperLOA - bias)) if not two_lines else 0),
        f'Mean: {"%.1f" % bias} {newline if two_lines else ""} [{"%.1f" % biasLowerCI}, {"%.1f" % biasUpperCI}]',
        ha="center",
        va="center",
    )
    ax.text(
        xText,
        upperLOA + ((0.4 * (upperLOA - bias)) if not two_lines else 0),
        f'+1.96SD: {"%.1f" % upperLOA} {newline if two_lines else ""} [{"%.1f" % upperLOA_lowerCI}, {"%.1f" % upperLOA_upperCI}]',
        ha="center",
        va="center",
    )
    print(f"p-value no correlation x-y: {corr_test(xs, diff)}.")
    ax.set_xlim(xlim)
    return diff


# deprecated: not accounting for repeated measures
def get_ba_stats_depr(x, y):
    current_dir = os.getcwd().replace("\\", "/")
    pd.DataFrame({"x": x, "y": y}).to_csv(current_dir + "/temp.csv")
    robjects.r(
        f"""
        library(blandr)
        library(rjson)

        df <- read.csv("{current_dir + "/temp.csv"}")
        stats <- blandr.statistics(df$x, df$y, sig.level=0.95)
        jsonStats <- toJSON(stats)
        write(
          jsonStats,
          "{current_dir + "/temp.json"}")
    """
    )
    with open(current_dir + "/temp.json", "r") as f:
        stats = json.load(f)
    os.remove(current_dir + "/temp.csv")
    os.remove(current_dir + "/temp.json")
    return stats


# deprecated: not accounting for repeated measures
def bland_altman_plot_depr(
    data1,
    data2,
    ax,
    percent=False,
    relative=False,
    x_true=False,
    bootstrapped=False,
    text_left=False,
    spacing=0.8,
    two_lines=True,
):
    mean = 0.5 * data1 + 0.5 * data2
    if x_true:
        xs = data1
        ax.set_xlabel("Measured [mmol]")
    else:
        xs = mean
        ax.set_xlabel("Mean (measured, predicted) [mmol]")
    if percent:
        data1 = data1 / mean
        data2 = data2 / mean
    elif relative:
        data2 = data2 / data1
        data1 = data1 / data1

    # have bias in the right direction
    ba_stats = get_ba_stats_depr(data2, data1)
    diff = np.asarray(ba_stats["differences"])
    # bias
    bias = ba_stats["bias"]
    # LOA
    lowerLOA = ba_stats["lowerLOA"]
    upperLOA = ba_stats["upperLOA"]
    # CIs
    if not bootstrapped:
        biasLowerCI = ba_stats["biasLowerCI"]
        biasUpperCI = ba_stats["biasUpperCI"]
        lowerLOA_lowerCI = ba_stats["lowerLOA_lowerCI"]
        lowerLOA_upperCI = ba_stats["lowerLOA_upperCI"]
        upperLOA_lowerCI = ba_stats["upperLOA_lowerCI"]
        upperLOA_upperCI = ba_stats["upperLOA_upperCI"]
    else:

        def lowerLOA_calc(*args):
            return bias - np.std(args) * 1.96

        def upperLOA_calc(*args):
            return bias + np.std(args) * 1.96

        bias_conf = bootstrap(
            (diff,),
            np.mean,
            confidence_level=0.95,
            vectorized=False,
            n_resamples=1e4,
            random_state=42,
        ).confidence_interval
        lowerLOA_conf = bootstrap(
            (diff,),
            lowerLOA_calc,
            confidence_level=0.95,
            vectorized=False,
            n_resamples=1e4,
            random_state=42,
        ).confidence_interval
        upperLOA_conf = bootstrap(
            (diff,),
            upperLOA_calc,
            confidence_level=0.95,
            vectorized=False,
            n_resamples=1e4,
            random_state=42,
        ).confidence_interval
        biasLowerCI = bias_conf.low
        biasUpperCI = bias_conf.high
        lowerLOA_lowerCI = lowerLOA_conf.low
        lowerLOA_upperCI = lowerLOA_conf.high
        upperLOA_lowerCI = upperLOA_conf.low
        upperLOA_upperCI = upperLOA_conf.high

    ax.scatter(xs, diff, s=1, alpha=0.2, c="k")
    # bias
    ax.axhline(0, c="k", linewidth=1)
    ax.axhline(bias, c="b", linewidth=1)
    xlim = ax.get_xlim()
    xlim = (0.0, xlim[1])
    ax.fill_between(
        [xlim[0], xlim[1] * 1.5],
        [biasUpperCI] * 2,
        [biasLowerCI] * 2,
        color="b",
        alpha=0.2,
    )
    # LOA
    ax.axhline(ba_stats["upperLOA"], c="r", linewidth=1)
    ax.axhline(lowerLOA, c="r", linewidth=1)
    # xlim hack for multiple plots...
    ax.fill_between(
        [xlim[0], xlim[1] * 1.5],
        [upperLOA_upperCI] * 2,
        [upperLOA_lowerCI] * 2,
        color="r",
        alpha=0.2,
    )
    ax.fill_between(
        [xlim[0], xlim[1] * 1.5],
        [lowerLOA_upperCI] * 2,
        [lowerLOA_lowerCI] * 2,
        color="r",
        alpha=0.2,
    )
    if bootstrapped:
        plot_regression(xs, diff, ax)
    if percent:
        ax.set_ylabel("Mean-normalized difference")
    elif relative:
        ax.set_ylabel("Relative difference \n((predicted-measured)/measured)")
    else:
        ax.set_ylabel("Absolute difference \n(predicted-measured) [mmol]")
    xText = 0.2 if text_left else xlim[1] * spacing
    newline = "\n"
    ax.text(
        xText,
        lowerLOA + ((0.3 * (upperLOA - bias)) if not two_lines else 0),
        f'-1.96SD: {"%.1f" % lowerLOA} {newline if two_lines else ""} [{"%.1f" % lowerLOA_lowerCI}, {"%.1f" % lowerLOA_upperCI}]',
        ha="center",
        va="center",
    )
    ax.text(
        xText,
        bias + ((0.4 * (upperLOA - bias)) if not two_lines else 0),
        f'Mean: {"%.1f" % bias} {newline if two_lines else ""} [{"%.1f" % biasLowerCI}, {"%.1f" % biasUpperCI}]',
        ha="center",
        va="center",
    )
    ax.text(
        xText,
        upperLOA + ((0.4 * (upperLOA - bias)) if not two_lines else 0),
        f'+1.96SD: {"%.1f" % upperLOA} {newline if two_lines else ""} [{"%.1f" % upperLOA_lowerCI}, {"%.1f" % upperLOA_upperCI}]',
        ha="center",
        va="center",
    )
    print(f"p-value no correlation x-y: {corr_test(xs, diff)}.")
    ax.set_xlim(xlim)
    return diff


def test_gaussian(data, alpha=0.05):
    print(f"Gaussianity tests ran with alpha={alpha}:")
    _, p_shapiro = shapiro(data)
    print(f"Shapiro-Wilk Test: p-value={p_shapiro}")
    if p_shapiro > alpha:
        print("==> Data looks Gaussian (fail to reject H0)")
    else:
        print("==> Data does not look Gaussian (reject H0)")
    _, p_dagostino = normaltest(data)
    print(f"D'Agostino's $K^2$ Test: p-value={p_dagostino}")
    if p_dagostino > alpha:
        print("==> Data looks Gaussian (fail to reject H0)")
    else:
        print("==> Data does not look Gaussian (reject H0)")
    result = anderson(data)
    print("Statistic: %.3f" % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print("%.3f: %.3f, data looks normal (fail to reject H0)" % (sl, cv))
        else:
            print("%.3f: %.3f, data does not look normal (reject H0)" % (sl, cv))
