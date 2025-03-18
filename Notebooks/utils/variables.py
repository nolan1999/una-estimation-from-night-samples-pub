corrected = True  # correct to 24-hour values
target_col = f"una_u24{'corr' if corrected else ''}_mmol"
min_pred_col = "una_un_mmol_norm_min" + ("_corr" if corrected else "")
vol_pred_col = "una_un_mmol_norm_vol" + ("_corr" if corrected else "")
min_pred_col_day = "una_ud_mmol_norm_min" + ("_corr" if corrected else "")
vol_pred_col_day = "una_ud_mmol_norm_vol" + ("_corr" if corrected else "")
mlr_pred_col = "mlr_preds"
