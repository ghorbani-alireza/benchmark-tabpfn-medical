import numpy as np
import warnings
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal, multivariate_t
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



# Simulation Function
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