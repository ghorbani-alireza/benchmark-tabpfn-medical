import numpy as np
import warnings
import time
from tqdm import tqdm
from scipy.stats import multivariate_normal, multivariate_t
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, balanced_accuracy_score, roc_auc_score, mean_squared_error
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
    run_exaone,
    run_tabicl
)
from . import utils 

#parallel functions are after this function

def _run_and_pack(fn, y_test, *args, **kwargs):
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start

    mse, auc, fpr, tpr, proba, brier = res[2], res[1], res[3], res[4], res[5], res[6]
    p = np.clip(proba, 0.001, 0.999)
    cal = LogisticRegression(solver='liblinear').fit(logit(p).reshape(-1, 1), y_test)

    return (elapsed, mse, auc, fpr, tpr, proba, brier,
            cal.coef_[0][0], cal.intercept_[0])


def _run_and_store(errors, prefix, fn, y_test, *args, **kwargs):
    t, mse, auc, fpr, tpr, proba, brier, slope, intercept = _run_and_pack(
        fn, y_test, *args, **kwargs)
    errors[f'{prefix}_Time'].append(t)
    errors[f'{prefix}_MSE'].append(mse)
    errors[f'{prefix}_AUC'].append(auc)
    errors[f'{prefix}_FPR'].append(fpr)
    errors[f'{prefix}_TPR'].append(tpr)
    errors[f'{prefix}_Brier'].append(brier)
    errors[f'{prefix}_CalSlope'].append(slope)
    errors[f'{prefix}_CalIntercept'].append(intercept)
    errors[f'{prefix}_Probas'].append(proba)

# Simulation Function - normal loop
def simulation(n, mu, sigma1, sigma2, df=None, dist="Normal", iter=100, g_seed=2025):
    n1_tr, n2_tr = n['n1_tr'], n['n2_tr']
    n1_te, n2_te = n['n1_te'], n['n2_te']
    mu1, mu2 = mu['mu1'], mu['mu2']

    np.random.seed(g_seed)

    errors = {
        'L-SLR_MSE': [], 'L-SLR_AUC': [], 'L-SLR_Time': [], 'L-SLR_FPR': [], 'L-SLR_TPR': [],
        'L-SLR_Brier': [], 'L-SLR_CalSlope': [], 'L-SLR_CalIntercept': [],
        'L-SLR_Probas': [],
        'SLR_MSE': [], 'SLR_AUC': [], 'SLR_Time': [], 'SLR_FPR': [], 'SLR_TPR': [],
        'SLR_Brier': [], 'SLR_CalSlope': [], 'SLR_CalIntercept': [],
        'SLR_Probas': [],
        'RandomForest_MSE': [], 'RandomForest_AUC': [], 'RandomForest_Time': [], 'RandomForest_FPR': [], 'RandomForest_TPR': [],
        'RandomForest_Brier': [], 'RandomForest_CalSlope': [], 'RandomForest_CalIntercept': [],
        'RandomForest_Probas': [],
        'XGBoost_MSE': [], 'XGBoost_AUC': [], 'XGBoost_Time': [], 'XGBoost_FPR': [], 'XGBoost_TPR': [],
        'XGBoost_Brier': [], 'XGBoost_CalSlope': [], 'XGBoost_CalIntercept': [],
        'XGBoost_Probas': [],
        'CatBoost_MSE': [], 'CatBoost_AUC': [], 'CatBoost_Time': [], 'CatBoost_FPR': [], 'CatBoost_TPR': [],
        'CatBoost_Brier': [], 'CatBoost_CalSlope': [], 'CatBoost_CalIntercept': [],
        'CatBoost_Probas': [],
        'TabPFN_v25_MSE': [], 'TabPFN_v25_AUC': [], 'TabPFN_v25_Time': [], 'TabPFN_v25_FPR': [], 'TabPFN_v25_TPR': [],
        'TabPFN_v25_Brier': [], 'TabPFN_v25_CalSlope': [], 'TabPFN_v25_CalIntercept': [],
        'TabPFN_v25_Probas': [],
        'TabPFN_v26_MSE': [], 'TabPFN_v26_AUC': [], 'TabPFN_v26_Time': [], 'TabPFN_v26_FPR': [], 'TabPFN_v26_TPR': [],
        'TabPFN_v26_Brier': [], 'TabPFN_v26_CalSlope': [], 'TabPFN_v26_CalIntercept': [],
        'TabPFN_v26_Probas': [],
        'TabPFN_v3_MSE': [], 'TabPFN_v3_AUC': [], 'TabPFN_v3_Time': [], 'TabPFN_v3_FPR': [], 'TabPFN_v3_TPR': [],
        'TabPFN_v3_Brier': [], 'TabPFN_v3_CalSlope': [], 'TabPFN_v3_CalIntercept': [],
        'TabPFN_v3_Probas': [],
        'TabPFN_v35_MSE': [], 'TabPFN_v35_AUC': [], 'TabPFN_v35_Time': [], 'TabPFN_v35_FPR': [], 'TabPFN_v35_TPR': [],
        'TabPFN_v35_Brier': [], 'TabPFN_v35_CalSlope': [], 'TabPFN_v35_CalIntercept': [],
        'TabPFN_v35_Probas': [],
        'EXAONE_MSE': [], 'EXAONE_AUC': [], 'EXAONE_Time': [], 'EXAONE_FPR': [], 'EXAONE_TPR': [],
        'EXAONE_Brier': [], 'EXAONE_CalSlope': [], 'EXAONE_CalIntercept': [],
        'EXAONE_Probas': [],
        'TabICL-V2_MSE': [], 'TabICL-V2_AUC': [], 'TabICL-V2_Time': [], 'TabICL-V2_FPR': [], 'TabICL-V2_TPR': [],
        'TabICL-V2_Brier': [], 'TabICL-V2_CalSlope': [], 'TabICL-V2_CalIntercept': [],
        'TabICL-V2_Probas': [],
        'y_test_all': []
    }
    
    captured_warnings = [] 

    for i in tqdm(range(iter), desc=f"Sim {dist} Iterations"):
        utils.current_iteration = i
        warnings.showwarning = utils.capture_warnings

        rng_tr = np.random.RandomState(i+ g_seed)
        rng_te = np.random.RandomState(i+ g_seed + 2000)

        if dist == "Normal":
            # generate from Normal
            Tr_x1 = rng_tr.multivariate_normal(mean=mu1, cov=sigma1, size=n1_tr)
            Tr_x2 = rng_tr.multivariate_normal(mean=mu2, cov=sigma2, size=n2_tr)
            Te_x1 = rng_te.multivariate_normal(mean=mu1, cov=sigma1, size=n1_te)
            Te_x2 = rng_te.multivariate_normal(mean=mu2, cov=sigma2, size=n2_te)
        elif dist == "t":
            # compute scale matrices
            if df <= 2:
                raise ValueError("Degrees of freedom must be > 2 for valid variance")
            scale1 = sigma1 * (df - 2) / df
            scale2 = sigma2 * (df - 2) / df
            # generate from t-dist
            Tr_x1 = multivariate_t.rvs(loc=mu1, shape=scale1, df=df, size=n1_tr, random_state=rng_tr)
            Tr_x2 = multivariate_t.rvs(loc=mu2, shape=scale2, df=df, size=n2_tr, random_state=rng_tr)
            Te_x1 = multivariate_t.rvs(loc=mu1, shape=scale1, df=df, size=n1_te, random_state=rng_te)
            Te_x2 = multivariate_t.rvs(loc=mu2, shape=scale2, df=df, size=n2_te, random_state=rng_te)
        else:
            raise ValueError("dist must be 'Normal' or 't'")

        X_train = np.vstack((Tr_x1, Tr_x2))
        y_train = np.array([0] * n1_tr + [1] * n2_tr)
        X_test = np.vstack((Te_x1, Te_x2))
        y_test = np.array([0] * n1_te + [1] * n2_te)

        # y_test for this iteration
        errors['y_test_all'].append(y_test)

        # Split training data (80/20)
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=i+g_seed
        )

        # run models and store

        _run_and_store(errors, 'L-SLR', run_linear_sparse_logistic_regression, y_test,
               X_train, X_test, y_train, y_test,
               X_train_sub, y_train_sub, X_val, y_val, random_state=i+g_seed)

        _run_and_store(errors, 'SLR', run_sparse_logistic_regression, y_test,
               X_train, X_test, y_train, y_test,
               X_train_sub, y_train_sub, X_val, y_val, random_state=i+g_seed)

        _run_and_store(errors, 'RandomForest', run_random_forest, y_test,
               X_train, X_test, y_train, y_test,
               X_train_sub, y_train_sub, X_val, y_val, random_state=i+g_seed)

        _run_and_store(errors, 'XGBoost', run_xgboost, y_test,
               X_train, X_test, y_train, y_test,
               X_train_sub, y_train_sub, X_val, y_val, random_state=i+g_seed)

        _run_and_store(errors, 'CatBoost', run_catboost, y_test, 
               X_train, X_test, y_train, y_test,
               X_train_sub, y_train_sub, X_val, y_val, random_state=i+g_seed)

        _run_and_store(errors, 'TabPFN_v25', run_tabpfn_v25, y_test,
               X_train, X_test, y_train, y_test, random_state=i+g_seed)

        _run_and_store(errors, 'TabPFN_v26', run_tabpfn_v26, y_test,
               X_train, X_test, y_train, y_test, random_state=i+g_seed)

        _run_and_store(errors, 'TabPFN_v3', run_tabpfn_v3, y_test,
               X_train, X_test, y_train, y_test, random_state=i+g_seed)

        _run_and_store(errors, 'TabPFN_v35', run_tabpfn_v35, y_test,
               X_train, X_test, y_train, y_test, random_state=i+g_seed)

        _run_and_store(errors, 'EXAONE', run_exaone, y_test,
               X_train, X_test, y_train, y_test, random_state=i+g_seed)
        
        _run_and_store(errors, 'TabICL-V2', run_tabicl, y_test,
               X_train, X_test, y_train, y_test, random_state=i+g_seed)

    return errors, captured_warnings
        



# parallel simulation function

# part 1 - function to evaluate on simulated data in parallel
def evaluate_on_sim_data(n, mu, sigma1, sigma2, df=None, dist="Normal", iter=100, g_seed=2025):
    n1_tr, n2_tr = n['n1_tr'], n['n2_tr']
    n1_te, n2_te = n['n1_te'], n['n2_te']
    mu1, mu2 = mu['mu1'], mu['mu2']

    np.random.seed(g_seed)

    errors = {
        'L-SLR_MSE': [], 'L-SLR_AUC': [], 'L-SLR_Time': [], 'L-SLR_FPR': [], 'L-SLR_TPR': [],
        'L-SLR_Brier': [], 'L-SLR_CalSlope': [], 'L-SLR_CalIntercept': [],
        'L-SLR_Probas': [],
        'SLR_MSE': [], 'SLR_AUC': [], 'SLR_Time': [], 'SLR_FPR': [], 'SLR_TPR': [],
        'SLR_Brier': [], 'SLR_CalSlope': [], 'SLR_CalIntercept': [],
        'SLR_Probas': [],
        'RandomForest_MSE': [], 'RandomForest_AUC': [], 'RandomForest_Time': [], 'RandomForest_FPR': [], 'RandomForest_TPR': [],
        'RandomForest_Brier': [], 'RandomForest_CalSlope': [], 'RandomForest_CalIntercept': [],
        'RandomForest_Probas': [],
        'XGBoost_MSE': [], 'XGBoost_AUC': [], 'XGBoost_Time': [], 'XGBoost_FPR': [], 'XGBoost_TPR': [],
        'XGBoost_Brier': [], 'XGBoost_CalSlope': [], 'XGBoost_CalIntercept': [],
        'XGBoost_Probas': [],
        'CatBoost_MSE': [], 'CatBoost_AUC': [], 'CatBoost_Time': [], 'CatBoost_FPR': [], 'CatBoost_TPR': [],
        'CatBoost_Brier': [], 'CatBoost_CalSlope': [], 'CatBoost_CalIntercept': [],
        'CatBoost_Probas': [],
        'TabPFN_v25_MSE': [], 'TabPFN_v25_AUC': [], 'TabPFN_v25_Time': [], 'TabPFN_v25_FPR': [], 'TabPFN_v25_TPR': [],
        'TabPFN_v25_Brier': [], 'TabPFN_v25_CalSlope': [], 'TabPFN_v25_CalIntercept': [],
        'TabPFN_v25_Probas': [],
        'TabPFN_v26_MSE': [], 'TabPFN_v26_AUC': [], 'TabPFN_v26_Time': [], 'TabPFN_v26_FPR': [], 'TabPFN_v26_TPR': [],
        'TabPFN_v26_Brier': [], 'TabPFN_v26_CalSlope': [], 'TabPFN_v26_CalIntercept': [],
        'TabPFN_v26_Probas': [],
        'TabPFN_v3_MSE': [], 'TabPFN_v3_AUC': [], 'TabPFN_v3_Time': [], 'TabPFN_v3_FPR': [], 'TabPFN_v3_TPR': [],
        'TabPFN_v3_Brier': [], 'TabPFN_v3_CalSlope': [], 'TabPFN_v3_CalIntercept': [],
        'TabPFN_v3_Probas': [],
        'TabPFN_v35_MSE': [], 'TabPFN_v35_AUC': [], 'TabPFN_v35_Time': [], 'TabPFN_v35_FPR': [], 'TabPFN_v35_TPR': [],
        'TabPFN_v35_Brier': [], 'TabPFN_v35_CalSlope': [], 'TabPFN_v35_CalIntercept': [],
        'TabPFN_v35_Probas': [],
        'EXAONE_MSE': [], 'EXAONE_AUC': [], 'EXAONE_Time': [], 'EXAONE_FPR': [], 'EXAONE_TPR': [],
        'EXAONE_Brier': [], 'EXAONE_CalSlope': [], 'EXAONE_CalIntercept': [],
        'EXAONE_Probas': [],
        'TabICL-V2_MSE': [], 'TabICL-V2_AUC': [], 'TabICL-V2_Time': [], 'TabICL-V2_FPR': [], 'TabICL-V2_TPR': [],
        'TabICL-V2_Brier': [], 'TabICL-V2_CalSlope': [], 'TabICL-V2_CalIntercept': [],
        'TabICL-V2_Probas': [],
        'y_test_all': []
    }
    
    
    # parallel - cahnge n_job for ibe
    results = Parallel(n_jobs=20, prefer="threads", verbose=0)(
        delayed(_single_sim_iteration)(
            i, n1_tr, n2_tr, n1_te, n2_te, mu1, mu2, sigma1, sigma2, df, dist, g_seed)
        for i in tqdm(range(iter), desc="Real‑data resampling", leave=True)
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
            ('TabPFN_v25', 'TabPFN_v25_'),
            ('TabPFN_v26', 'TabPFN_v26_'),
            ('TabPFN_v3', 'TabPFN_v3_'),
            ('TabPFN_v35', 'TabPFN_v35_'),
            ('EXAONE', 'EXAONE_'),
            ('TabICL-V2', 'TabICL-V2_')
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
    
    
# part2 - fucntion for single iteration
def _single_sim_iteration(i, n1_tr, n2_tr, n1_te, n2_te, mu1, mu2,
                          sigma1, sigma2, df, dist, g_seed):
    
    np.random.seed(g_seed)
    
    rng_tr = np.random.RandomState(i+ g_seed)
    rng_te = np.random.RandomState(i+ g_seed + 2000)

    if dist == "Normal":
        # generate from Normal
        Tr_x1 = rng_tr.multivariate_normal(mean=mu1, cov=sigma1, size=n1_tr)
        Tr_x2 = rng_tr.multivariate_normal(mean=mu2, cov=sigma2, size=n2_tr)
        Te_x1 = rng_te.multivariate_normal(mean=mu1, cov=sigma1, size=n1_te)
        Te_x2 = rng_te.multivariate_normal(mean=mu2, cov=sigma2, size=n2_te)
    elif dist == "t":
        # compute scale matrices
        if df is None or df <= 2:
            raise ValueError("Degrees of freedom must be > 2 for valid variance")
        scale1 = sigma1 * (df - 2) / df
        scale2 = sigma2 * (df - 2) / df
        # generate from t-dist
        Tr_x1 = multivariate_t.rvs(loc=mu1, shape=scale1, df=df, size=n1_tr, random_state=rng_tr)
        Tr_x2 = multivariate_t.rvs(loc=mu2, shape=scale2, df=df, size=n2_tr, random_state=rng_tr)
        Te_x1 = multivariate_t.rvs(loc=mu1, shape=scale1, df=df, size=n1_te, random_state=rng_te)
        Te_x2 = multivariate_t.rvs(loc=mu2, shape=scale2, df=df, size=n2_te, random_state=rng_te)
    else:
        raise ValueError("dist must be 'Normal' or 't'")

    X_train = np.vstack((Tr_x1, Tr_x2))
    y_train = np.array([0] * n1_tr + [1] * n2_tr)
    X_test = np.vstack((Te_x1, Te_x2))
    y_test = np.array([0] * n1_te + [1] * n2_te)

    # Split training data (80/20)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=i+g_seed
    )

    # run models and store
    
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
        'EXAONE': _run_and_pack(
            run_exaone, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed),
        'TabICL-V2': _run_and_pack(
            run_tabicl, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed)
    }



