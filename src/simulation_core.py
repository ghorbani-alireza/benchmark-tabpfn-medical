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
    run_tabpfn
)
from . import utils 

#parallel functions are after this function

# Simulation Function - normal loop
def simulation(n, mu, sigma1, sigma2, df=5, dist="Normal", iter=100, g_seed=2025):
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
        'TabPFN_MSE': [], 'TabPFN_AUC': [], 'TabPFN_Time': [], 'TabPFN_FPR': [], 'TabPFN_TPR': [],
        'TabPFN_Brier': [], 'TabPFN_CalSlope': [], 'TabPFN_CalIntercept': [],
        'TabPFN_Probas': [],
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


        # L-SLR
        start_time = time.perf_counter()
        logreg_result = run_linear_sparse_logistic_regression(X_train, X_test,
                                y_train, y_test, X_train_sub, y_train_sub,
                                X_val, y_val, random_state=i+g_seed)
        errors['L-SLR_Time'].append(time.perf_counter() - start_time)
        errors['L-SLR_MSE'].append(logreg_result[2])
        errors['L-SLR_AUC'].append(logreg_result[1])
        errors['L-SLR_FPR'].append(logreg_result[3])
        errors['L-SLR_TPR'].append(logreg_result[4])
        proba = logreg_result[5]
        errors['L-SLR_Probas'].append(proba)
        errors['L-SLR_Brier'].append(logreg_result[6])
        proba_clipped = np.clip(proba, 0.001, 0.999)
        logit_proba = logit(proba_clipped)
        cal_model = LogisticRegression(solver='liblinear')
        cal_model.fit(logit_proba.reshape(-1, 1), y_test)
        errors['L-SLR_CalSlope'].append(cal_model.coef_[0][0])
        errors['L-SLR_CalIntercept'].append(cal_model.intercept_[0])

        # SLR
        start_time = time.perf_counter()
        auglogreg_result = run_sparse_logistic_regression(X_train, X_test,
                                y_train, y_test, X_train_sub, y_train_sub,
                                X_val, y_val, random_state=i+g_seed)
        errors['SLR_Time'].append(time.perf_counter() - start_time)
        errors['SLR_MSE'].append(auglogreg_result[2])
        errors['SLR_AUC'].append(auglogreg_result[1])
        errors['SLR_FPR'].append(auglogreg_result[3])
        errors['SLR_TPR'].append(auglogreg_result[4])
        proba = auglogreg_result[5]
        errors['SLR_Probas'].append(proba)
        errors['SLR_Brier'].append(auglogreg_result[6])
        proba_clipped = np.clip(proba, 0.001, 0.999)
        logit_proba = logit(proba_clipped)
        cal_model = LogisticRegression(solver='liblinear')
        cal_model.fit(logit_proba.reshape(-1, 1), y_test)
        errors['SLR_CalSlope'].append(cal_model.coef_[0][0])
        errors['SLR_CalIntercept'].append(cal_model.intercept_[0])

        # RF
        start_time = time.perf_counter()
        rf_result = run_random_forest(X_train, X_test, y_train, y_test,
                                      X_train_sub, y_train_sub, X_val, y_val,
                                      random_state=i+g_seed)
        errors['RandomForest_Time'].append(time.perf_counter() - start_time)
        errors['RandomForest_MSE'].append(rf_result[2])
        errors['RandomForest_AUC'].append(rf_result[1])
        errors['RandomForest_FPR'].append(rf_result[3])
        errors['RandomForest_TPR'].append(rf_result[4])
        proba = rf_result[5]
        errors['RandomForest_Probas'].append(proba)
        errors['RandomForest_Brier'].append(rf_result[6])
        proba_clipped = np.clip(proba, 0.001, 0.999)
        logit_proba = logit(proba_clipped)
        cal_model = LogisticRegression(solver='liblinear')
        cal_model.fit(logit_proba.reshape(-1, 1), y_test)
        errors['RandomForest_CalSlope'].append(cal_model.coef_[0][0])
        errors['RandomForest_CalIntercept'].append(cal_model.intercept_[0])

        # XGBoost
        start_time = time.perf_counter()
        xgb_result = run_xgboost(X_train, X_test, y_train, y_test, X_train_sub,
                                 y_train_sub, X_val, y_val,
                                 random_state=i+g_seed)
        errors['XGBoost_Time'].append(time.perf_counter() - start_time)
        errors['XGBoost_MSE'].append(xgb_result[2])
        errors['XGBoost_AUC'].append(xgb_result[1])
        errors['XGBoost_FPR'].append(xgb_result[3])
        errors['XGBoost_TPR'].append(xgb_result[4])
        proba = xgb_result[5]
        errors['XGBoost_Probas'].append(proba)
        errors['XGBoost_Brier'].append(xgb_result[6])
        proba_clipped = np.clip(proba, 0.001, 0.999)
        logit_proba = logit(proba_clipped)
        cal_model = LogisticRegression(solver='liblinear')
        cal_model.fit(logit_proba.reshape(-1, 1), y_test)
        errors['XGBoost_CalSlope'].append(cal_model.coef_[0][0])
        errors['XGBoost_CalIntercept'].append(cal_model.intercept_[0])

        # CatBoost
        start_time = time.perf_counter()
        cb_result = run_catboost(X_train, X_test, y_train, y_test, X_train_sub,
                                 y_train_sub, X_val, y_val,
                                 random_state=i+g_seed)
        errors['CatBoost_Time'].append(time.perf_counter() - start_time)
        errors['CatBoost_MSE'].append(cb_result[2])
        errors['CatBoost_AUC'].append(cb_result[1])
        errors['CatBoost_FPR'].append(cb_result[3])
        errors['CatBoost_TPR'].append(cb_result[4])
        proba = cb_result[5]
        errors['CatBoost_Probas'].append(proba)
        errors['CatBoost_Brier'].append(cb_result[6])
        proba_clipped = np.clip(proba, 0.001, 0.999)
        logit_proba = logit(proba_clipped)
        cal_model = LogisticRegression(solver='liblinear')
        cal_model.fit(logit_proba.reshape(-1, 1), y_test)
        errors['CatBoost_CalSlope'].append(cal_model.coef_[0][0])
        errors['CatBoost_CalIntercept'].append(cal_model.intercept_[0])

        # TabPFN
        start_time = time.perf_counter()
        tabpfn_result = run_tabpfn(X_train, X_test, y_train, y_test,
                                   random_state=i+g_seed)
        errors['TabPFN_Time'].append(time.perf_counter() - start_time)
        errors['TabPFN_MSE'].append(tabpfn_result[2])
        errors['TabPFN_AUC'].append(tabpfn_result[1])
        errors['TabPFN_FPR'].append(tabpfn_result[3])
        errors['TabPFN_TPR'].append(tabpfn_result[4])
        proba = tabpfn_result[5]
        errors['TabPFN_Probas'].append(proba)
        errors['TabPFN_Brier'].append(tabpfn_result[6])
        proba_clipped = np.clip(proba, 0.001, 0.999)
        logit_proba = logit(proba_clipped)
        cal_model = LogisticRegression(solver='liblinear')
        cal_model.fit(logit_proba.reshape(-1, 1), y_test)
        errors['TabPFN_CalSlope'].append(cal_model.coef_[0][0])
        errors['TabPFN_CalIntercept'].append(cal_model.intercept_[0])

    return errors, captured_warnings   



# parallel simulation function

# part 1 - function to evaluate on simulated data in parallel
def evaluate_on_sim_data(n, mu, sigma1, sigma2, df=5, dist="Normal", iter=100, g_seed=2025):
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
        'TabPFN_MSE': [], 'TabPFN_AUC': [], 'TabPFN_Time': [], 'TabPFN_FPR': [], 'TabPFN_TPR': [],
        'TabPFN_Brier': [], 'TabPFN_CalSlope': [], 'TabPFN_CalIntercept': [],
        'TabPFN_Probas': [],
        'y_test_all': []
    }
    
    
    # parallel - cahnge n_job for ibe
    results = Parallel(n_jobs=10, prefer="threads", verbose=0)(
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

    # Split training data (80/20)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=i+g_seed
    )

    # run models and store
    
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



