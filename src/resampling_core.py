import time
from tqdm import tqdm
import numpy as np

from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# import models from model_definitions.py
from .model_definitions import (
    run_linear_sparse_logistic_regression,
    run_sparse_logistic_regression,
    run_random_forest,
    run_xgboost,
    run_catboost,
    run_tabpfn
)


# resampling Functions for realdata

# part2 - fucntion for single iteration calcualtion
def _single_real_iteration(i, X_np, y_np, idx0, idx1,
                           n1_tr, n2_tr, n1_te, n2_te, g_seed):

    np.random.seed(g_seed)
    rng = np.random.default_rng(seed=i+g_seed)

    train0 = rng.choice(idx0, size=n1_tr, replace=False)
    train1 = rng.choice(idx1, size=n2_tr, replace=False)
    remaining0 = np.setdiff1d(idx0, train0, assume_unique=True) # separate test
    remaining1 = np.setdiff1d(idx1, train1, assume_unique=True)
    test0 = rng.choice(remaining0, size=n1_te, replace=False)
    test1 = rng.choice(remaining1, size=n2_te, replace=False)

    train_idx = np.concatenate([train0, train1])
    test_idx = np.concatenate([test0, test1])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = X_np[train_idx]
    y_train = y_np[train_idx]
    X_test = X_np[test_idx]
    y_test = y_np[test_idx]

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=i+g_seed
    )

    # L-SLR
    start = time.perf_counter()
    res = run_linear_sparse_logistic_regression(X_train, X_test, y_train, y_test,
                                                 X_train_sub, y_train_sub, X_val,
                                                y_val, random_state=i+g_seed)
    lslr_time = time.perf_counter() - start
    lslr_mse, lslr_auc, lslr_fpr, lslr_tpr, lslr_proba, lslr_brier = res[2], res[1], res[3], res[4], res[5], res[6]

    lslr_proba_clipped = np.clip(lslr_proba, 0.001, 0.999)
    lslr_logit = logit(lslr_proba_clipped)
    lslr_cal = LogisticRegression(solver='liblinear')
    lslr_cal.fit(lslr_logit.reshape(-1, 1), y_test)
    lslr_slope = lslr_cal.coef_[0][0]
    lslr_intercept = lslr_cal.intercept_[0]


    # SLR
    start = time.perf_counter()
    res = run_sparse_logistic_regression(X_train, X_test, y_train, y_test,
                                                 X_train_sub, y_train_sub, X_val,
                                                y_val, random_state=i+g_seed)
    slr_time = time.perf_counter() - start
    slr_mse, slr_auc, slr_fpr, slr_tpr, slr_proba, slr_brier = res[2], res[1], res[3], res[4], res[5], res[6]

    slr_proba_clipped = np.clip(slr_proba, 0.001, 0.999)
    slr_logit = logit(slr_proba_clipped)
    slr_cal = LogisticRegression(solver='liblinear')
    slr_cal.fit(slr_logit.reshape(-1, 1), y_test)
    slr_slope = slr_cal.coef_[0][0]
    slr_intercept = slr_cal.intercept_[0]

    # random forest
    start = time.perf_counter()
    res = run_random_forest(X_train, X_test, y_train, y_test,
                            X_train_sub, y_train_sub, X_val, y_val,
                            random_state=i+g_seed)
    rf_time = time.perf_counter() - start
    rf_mse, rf_auc, rf_fpr, rf_tpr, rf_proba, rf_brier = res[2], res[1], res[3], res[4], res[5], res[6]

    rf_proba_clipped = np.clip(rf_proba, 0.001, 0.999)
    rf_logit = logit(rf_proba_clipped)
    rf_cal = LogisticRegression(solver='liblinear')
    rf_cal.fit(rf_logit.reshape(-1, 1), y_test)
    rf_slope = rf_cal.coef_[0][0]
    rf_intercept = rf_cal.intercept_[0]

    # XGBoost
    start = time.perf_counter()
    res = run_xgboost(X_train, X_test, y_train, y_test,
                      X_train_sub, y_train_sub, X_val, y_val,
                      random_state=i+g_seed)
    xgb_time = time.perf_counter() - start
    xgb_mse, xgb_auc, xgb_fpr, xgb_tpr, xgb_proba, xgb_brier = res[2], res[1], res[3], res[4], res[5], res[6]

    xgb_proba_clipped = np.clip(xgb_proba, 0.001, 0.999)
    xgb_logit = logit(xgb_proba_clipped)
    xgb_cal = LogisticRegression(solver='liblinear')
    xgb_cal.fit(xgb_logit.reshape(-1, 1), y_test)
    xgb_slope = xgb_cal.coef_[0][0]
    xgb_intercept = xgb_cal.intercept_[0]

    #  CatBoost
    start = time.perf_counter()
    res = run_catboost(X_train, X_test, y_train, y_test,
                       X_train_sub, y_train_sub, X_val, y_val,
                       random_state=i+g_seed)
    cb_time = time.perf_counter() - start
    cb_mse, cb_auc, cb_fpr, cb_tpr, cb_proba, cb_brier = res[2], res[1], res[3], res[4], res[5], res[6]

    cb_proba_clipped = np.clip(cb_proba, 0.001, 0.999)
    cb_logit = logit(cb_proba_clipped)
    cb_cal = LogisticRegression(solver='liblinear')
    cb_cal.fit(cb_logit.reshape(-1, 1), y_test)
    cb_slope = cb_cal.coef_[0][0]
    cb_intercept = cb_cal.intercept_[0]

    # tabpfn
    start = time.perf_counter()
    res = run_tabpfn(X_train, X_test, y_train, y_test, random_state=i+g_seed)
    tpf_time = time.perf_counter() - start
    tpf_mse, tpf_auc, tpf_fpr, tpf_tpr, tpf_proba, tpf_brier = res[2], res[1], res[3], res[4], res[5], res[6]

    tpf_proba_clipped = np.clip(tpf_proba, 0.001, 0.999)
    tpf_logit = logit(tpf_proba_clipped)
    tpf_cal = LogisticRegression(solver='liblinear')
    tpf_cal.fit(tpf_logit.reshape(-1, 1), y_test)
    tpf_slope = tpf_cal.coef_[0][0]
    tpf_intercept = tpf_cal.intercept_[0]

    return {
        'y_test': y_test,
        'L-SLR': (lslr_time, lslr_mse, lslr_auc, lslr_fpr, lslr_tpr, lslr_proba, lslr_brier, lslr_slope, lslr_intercept),
        'SLR': (slr_time, slr_mse, slr_auc, slr_fpr, slr_tpr, slr_proba, slr_brier, slr_slope, slr_intercept),
        'RandomForest': (rf_time, rf_mse, rf_auc, rf_fpr, rf_tpr, rf_proba, rf_brier, rf_slope, rf_intercept),
        'XGBoost': (xgb_time, xgb_mse, xgb_auc, xgb_fpr, xgb_tpr, xgb_proba, xgb_brier, xgb_slope, xgb_intercept),
        'CatBoost': (cb_time, cb_mse, cb_auc, cb_fpr, cb_tpr, cb_proba, cb_brier, cb_slope, cb_intercept),
        'TabPFN': (tpf_time, tpf_mse, tpf_auc, tpf_fpr, tpf_tpr, tpf_proba, tpf_brier, tpf_slope, tpf_intercept),
    }




#  part2 - fucntion for all iterations
def evaluate_on_real_data(X, y, n, iter=100, g_seed = 2025):
    #np.random.seed(g_seed)

    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y, dtype=int)

    idx0 = np.where(y_np == 0)[0]
    idx1 = np.where(y_np == 1)[0]

    n1_tr, n2_tr = n['n1_tr'], n['n2_tr']
    n1_te, n2_te = n['n1_te'], n['n2_te']

    if len(idx0) < n1_tr + n1_te or len(idx1) < n2_tr + n2_te:
        raise ValueError(
            f"Not enough samples per class.\n"
            f"Class 0: have {len(idx0)}, need {n1_tr + n1_te}\n"
            f"Class 1: have {len(idx1)}, need {n2_tr + n2_te}"
        )

    errors = {
        'L-SLR_MSE': [], 'L-SLR_AUC': [], 'L-SLR_Time': [], 'L-SLR_FPR': [], 'L-SLR_TPR': [],
        'L-SLR_Brier': [], 'L-SLR_CalSlope': [], 'L-SLR_CalIntercept': [], 'L-SLR_Probas': [],
        'SLR_MSE': [], 'SLR_AUC': [], 'SLR_Time': [], 'SLR_FPR': [], 'SLR_TPR': [],
        'SLR_Brier': [], 'SLR_CalSlope': [], 'SLR_CalIntercept': [], 'SLR_Probas': [],
        'RandomForest_MSE': [], 'RandomForest_AUC': [], 'RandomForest_Time': [], 'RandomForest_FPR': [], 'RandomForest_TPR': [],
        'RandomForest_Brier': [], 'RandomForest_CalSlope': [], 'RandomForest_CalIntercept': [], 'RandomForest_Probas': [],
        'XGBoost_MSE': [], 'XGBoost_AUC': [], 'XGBoost_Time': [], 'XGBoost_FPR': [], 'XGBoost_TPR': [],
        'XGBoost_Brier': [], 'XGBoost_CalSlope': [], 'XGBoost_CalIntercept': [], 'XGBoost_Probas': [],
        'CatBoost_MSE': [], 'CatBoost_AUC': [], 'CatBoost_Time': [], 'CatBoost_FPR': [], 'CatBoost_TPR': [],
        'CatBoost_Brier': [], 'CatBoost_CalSlope': [], 'CatBoost_CalIntercept': [], 'CatBoost_Probas': [],
        'TabPFN_MSE': [], 'TabPFN_AUC': [], 'TabPFN_Time': [], 'TabPFN_FPR': [], 'TabPFN_TPR': [],
        'TabPFN_Brier': [], 'TabPFN_CalSlope': [], 'TabPFN_CalIntercept': [], 'TabPFN_Probas': [],
        'y_test_all': []
    }

    # parallel - cahnge n_job for ibe
    results = Parallel(n_jobs=10, prefer="threads", verbose=0)(
        delayed(_single_real_iteration)(
            i, X_np, y_np, idx0, idx1,
            n1_tr, n2_tr, n1_te, n2_te, g_seed
        )
        for i in tqdm(range(iter), desc="Real‑data resampling",
                                     leave=True)
    )
    print("parallel done - aggregating results...")
    t0 = time.perf_counter()

    for res in results:
        errors['y_test_all'].append(res['y_test'])
        for key, prefix in [
            ('L-SLR', 'L-SLR_'),
            ('SLR', 'SLR_'),
            ('RandomForest', 'RandomForest_'),
            ('XGBoost', 'XGBoost_'),
            ('CatBoost', 'CatBoost_'),
            ('TabPFN', 'TabPFN_')
        ]:
            (t, mse, auc, fpr, tpr, proba, brier, slope, intercept) = res[key]
            errors[prefix + 'Time'].append(t)
            errors[prefix + 'MSE'].append(mse)
            errors[prefix + 'AUC'].append(auc)
            errors[prefix + 'FPR'].append(fpr)
            errors[prefix + 'TPR'].append(tpr)
            errors[prefix + 'Brier'].append(brier)
            errors[prefix + 'CalSlope'].append(slope)
            errors[prefix + 'CalIntercept'].append(intercept)
            errors[prefix + 'Probas'].append(proba)
    print(f"aggregation time: {time.perf_counter() - t0:.2f} seconds")
    return errors



