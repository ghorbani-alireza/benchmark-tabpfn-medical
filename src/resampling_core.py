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
    run_tabpfn_v25,
    run_tabpfn_v26,
    run_tabpfn_v3,
    run_tabpfn_v35,
    run_mitra_v2,
    run_tabfm
)


# resampling Functions for realdata

def _run_and_pack(fn, y_test, *args, **kwargs):
    """Run a model fn, time it, compute calibration, return the 9-tuple."""
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start

    mse, auc, fpr, tpr, proba, brier = res[2], res[1], res[3], res[4], res[5], res[6]
    p = np.clip(proba, 0.001, 0.999)
    cal = LogisticRegression(solver='liblinear').fit(logit(p).reshape(-1, 1), y_test)

    return (elapsed, mse, auc, fpr, tpr, proba, brier,
            cal.coef_[0][0], cal.intercept_[0])

# part1 - fucntion for single iteration calcualtion
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

    return {
    'y_test': y_test,
    'L-SLR': _run_and_pack(
        run_linear_sparse_logistic_regression, y_test,
        X_train, X_test, y_train, y_test,
        X_train_sub, y_train_sub, X_val, y_val,
        random_state=i+g_seed),
    'SLR': _run_and_pack(
        run_sparse_logistic_regression, y_test,
        X_train, X_test, y_train, y_test,
        X_train_sub, y_train_sub, X_val, y_val,
        random_state=i+g_seed),
    'RandomForest': _run_and_pack(
        run_random_forest, y_test,
        X_train, X_test, y_train, y_test,
        X_train_sub, y_train_sub, X_val, y_val,
        random_state=i+g_seed),
    'XGBoost': _run_and_pack(
        run_xgboost, y_test,
        X_train, X_test, y_train, y_test,
        X_train_sub, y_train_sub, X_val, y_val,
        random_state=i+g_seed),
    'CatBoost': _run_and_pack(
        run_catboost, y_test,
        X_train, X_test, y_train, y_test,
        X_train_sub, y_train_sub, X_val, y_val,
        random_state=i+g_seed),
    'TabPFN_25': _run_and_pack(
        run_tabpfn_v25, y_test,
        X_train, X_test, y_train, y_test, random_state=i+g_seed),
    'TabPFN_v26': _run_and_pack(
        run_tabpfn_v26, y_test,
        X_train, X_test, y_train, y_test, random_state=i+g_seed),
    'TabPFN_v3': _run_and_pack(
        run_tabpfn_v3, y_test,
        X_train, X_test, y_train, y_test, random_state=i+g_seed),
    'TabPFN_v35': _run_and_pack(
        run_tabpfn_v35, y_test,
        X_train, X_test, y_train, y_test, random_state=i+g_seed),
    'Mitra-V2': _run_and_pack(
        run_mitra_v2, y_test,
        X_train, X_test, y_train, y_test, random_state=i+g_seed),
    'TabFM': _run_and_pack(
        run_tabfm, y_test,
        X_train, X_test, y_train, y_test, random_state=i+g_seed),
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
        'TabPFN_25_MSE': [], 'TabPFN_25_AUC': [], 'TabPFN_25_Time': [], 'TabPFN_25_FPR': [], 'TabPFN_25_TPR': [],
        'TabPFN_25_Brier': [], 'TabPFN_25_CalSlope': [], 'TabPFN_25_CalIntercept': [], 'TabPFN_25_Probas': [],
        'TabPFN_v26_MSE': [], 'TabPFN_v26_AUC': [], 'TabPFN_v26_Time': [], 'TabPFN_v26_FPR': [], 'TabPFN_v26_TPR': [],
        'TabPFN_v26_Brier': [], 'TabPFN_v26_CalSlope': [], 'TabPFN_v26_CalIntercept': [], 'TabPFN_v26_Probas': [],
        'TabPFN_v3_MSE': [], 'TabPFN_v3_AUC': [], 'TabPFN_v3_Time': [], 'TabPFN_v3_FPR': [], 'TabPFN_v3_TPR': [],
        'TabPFN_v3_Brier': [], 'TabPFN_v3_CalSlope': [], 'TabPFN_v3_CalIntercept': [], 'TabPFN_v3_Probas': [],
        'TabPFN_v35_MSE': [], 'TabPFN_v35_AUC': [], 'TabPFN_v35_Time': [], 'TabPFN_v35_FPR': [], 'TabPFN_v35_TPR': [],
        'TabPFN_v35_Brier': [], 'TabPFN_v35_CalSlope': [], 'TabPFN_v35_CalIntercept': [], 'TabPFN_v35_Probas': [],
        'Mitra-V2_MSE': [], 'Mitra-V2_AUC': [], 'Mitra-V2_Time': [], 'Mitra-V2_FPR': [], 'Mitra-V2_TPR': [],
        'Mitra-V2_Brier': [], 'Mitra-V2_CalSlope': [], 'Mitra-V2_CalIntercept': [], 'Mitra-V2_Probas': [],
        'TabFM_MSE': [], 'TabFM_AUC': [], 'TabFM_Time': [], 'TabFM_FPR': [], 'TabFM_TPR': [],
        'TabFM_Brier': [], 'TabFM_CalSlope': [], 'TabFM_CalIntercept': [], 'TabFM_Probas': [],
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
            ('TabPFN_25', 'TabPFN_25_'),
            ('TabPFN_v26', 'TabPFN_v26_'),
            ('TabPFN_v3', 'TabPFN_v3_'),
            ('TabPFN_v35', 'TabPFN_v35_'),
            ('Mitra-V2', 'Mitra-V2_'),
            ('TabFM', 'TabFM_')
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



