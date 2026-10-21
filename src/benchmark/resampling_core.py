import time
from tqdm import tqdm
import numpy as np

from joblib import Parallel, delayed, parallel_config
from pathlib import Path
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .utils import capture_output
from .execution import initialize_worker, run_logged_iteration
from . import config


# import models from model_definitions.py
from .model_definitions import (
    run_linear_sparse_logistic_regression,
    run_sparse_logistic_regression,
    run_random_forest,
    run_xgboost,
    run_catboost,
    run_quadro,
    run_tabpfn_v25,
    run_tabpfn_v26,
    run_tabpfn_v3,
    run_tabpfn_v35,
    run_exaone,
    run_tabicl,
)




def _run_and_pack(fn, y_test, *args, **kwargs):
    """Run one model, measure its runtime, and calculate calibration from its test probabilities."""
    
    # time all
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start

    (_, auc, mse, fpr, tpr, proba, brier,
     setup_time, tune_time, fit_time, predict_time) = res
    p = np.clip(proba, 0.001, 0.999)
    cal = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000).fit(logit(p).reshape(-1, 1), y_test)

    return (elapsed, mse, auc, fpr, tpr, proba, brier,
            cal.coef_[0][0], cal.intercept_[0],
            setup_time, tune_time, fit_time, predict_time)
    
 
 # resampling Functions for realdata

# part1 - fucntion for single iteration calcualtion
def _single_real_iteration(i, X_np, y_np, idx0, idx1,
                           n1_tr, n2_tr, n1_te, n2_te, g_seed, include_exaone=False):

    """Sample separate training and test rows from each class and evaluate every model once.
    
    The split is by row, not by patient. An inner training/validation split is used for tuning.
    """
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

    
    # run models and store
    
    results = {
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
        'Quadro': _run_and_pack(
            run_quadro, y_test,
            X_train, X_test, y_train, y_test,
            X_train_sub, y_train_sub, X_val, y_val,
            random_state=i+g_seed),
        'TabPFN_v25': _run_and_pack(
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
        'TabICL-V2': _run_and_pack(
            run_tabicl, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed),
        
    }
    if include_exaone:
        results['EXAONE'] = _run_and_pack(
            run_exaone, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed,
        )
    return results





#  part2 - fucntion for all iterations
def evaluate_on_real_data(X, y, n, iter=100, g_seed = 2025, *, include_exaone=False, warnings_file=None, n_jobs=10, progress_label=None):
    #np.random.seed(g_seed)

    """Repeat row-level sampling and model evaluation on an existing dataset.
    
    Check the available class sizes first, run repetitions in parallel, and combine their results.
    progress_label optionally identifies the dataset in the progress bar.
    """
    initialize_worker()
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
        'L-SLR_SetupTime': [], 'L-SLR_TuneTime': [], 'L-SLR_FitTime': [], 'L-SLR_PredictTime': [],
        'L-SLR_Brier': [], 'L-SLR_CalSlope': [], 'L-SLR_CalIntercept': [], 'L-SLR_Probas': [],
        'SLR_MSE': [], 'SLR_AUC': [], 'SLR_Time': [], 'SLR_FPR': [], 'SLR_TPR': [],
        'SLR_SetupTime': [], 'SLR_TuneTime': [], 'SLR_FitTime': [], 'SLR_PredictTime': [],
        'SLR_Brier': [], 'SLR_CalSlope': [], 'SLR_CalIntercept': [], 'SLR_Probas': [],
        'RandomForest_MSE': [], 'RandomForest_AUC': [], 'RandomForest_Time': [], 'RandomForest_FPR': [], 'RandomForest_TPR': [],
        'RandomForest_SetupTime': [], 'RandomForest_TuneTime': [], 'RandomForest_FitTime': [], 'RandomForest_PredictTime': [],
        'RandomForest_Brier': [], 'RandomForest_CalSlope': [], 'RandomForest_CalIntercept': [], 'RandomForest_Probas': [],
        'XGBoost_MSE': [], 'XGBoost_AUC': [], 'XGBoost_Time': [], 'XGBoost_FPR': [], 'XGBoost_TPR': [],
        'XGBoost_SetupTime': [], 'XGBoost_TuneTime': [], 'XGBoost_FitTime': [], 'XGBoost_PredictTime': [],
        'XGBoost_Brier': [], 'XGBoost_CalSlope': [], 'XGBoost_CalIntercept': [], 'XGBoost_Probas': [],
        'CatBoost_MSE': [], 'CatBoost_AUC': [], 'CatBoost_Time': [], 'CatBoost_FPR': [], 'CatBoost_TPR': [],
        'CatBoost_SetupTime': [], 'CatBoost_TuneTime': [], 'CatBoost_FitTime': [], 'CatBoost_PredictTime': [],
        'CatBoost_Brier': [], 'CatBoost_CalSlope': [], 'CatBoost_CalIntercept': [], 'CatBoost_Probas': [],
        'Quadro_MSE': [], 'Quadro_AUC': [], 'Quadro_Time': [], 'Quadro_FPR': [], 'Quadro_TPR': [],
        'Quadro_SetupTime': [], 'Quadro_TuneTime': [], 'Quadro_FitTime': [], 'Quadro_PredictTime': [],
        'Quadro_Brier': [], 'Quadro_CalSlope': [], 'Quadro_CalIntercept': [], 'Quadro_Probas': [],
        'TabPFN_v25_MSE': [], 'TabPFN_v25_AUC': [], 'TabPFN_v25_Time': [], 'TabPFN_v25_FPR': [], 'TabPFN_v25_TPR': [],
        'TabPFN_v25_SetupTime': [], 'TabPFN_v25_TuneTime': [], 'TabPFN_v25_FitTime': [], 'TabPFN_v25_PredictTime': [],
        'TabPFN_v25_Brier': [], 'TabPFN_v25_CalSlope': [], 'TabPFN_v25_CalIntercept': [], 'TabPFN_v25_Probas': [],
        'TabPFN_v26_MSE': [], 'TabPFN_v26_AUC': [], 'TabPFN_v26_Time': [], 'TabPFN_v26_FPR': [], 'TabPFN_v26_TPR': [],
        'TabPFN_v26_SetupTime': [], 'TabPFN_v26_TuneTime': [], 'TabPFN_v26_FitTime': [], 'TabPFN_v26_PredictTime': [],
        'TabPFN_v26_Brier': [], 'TabPFN_v26_CalSlope': [], 'TabPFN_v26_CalIntercept': [], 'TabPFN_v26_Probas': [],
        'TabPFN_v3_MSE': [], 'TabPFN_v3_AUC': [], 'TabPFN_v3_Time': [], 'TabPFN_v3_FPR': [], 'TabPFN_v3_TPR': [],
        'TabPFN_v3_SetupTime': [], 'TabPFN_v3_TuneTime': [], 'TabPFN_v3_FitTime': [], 'TabPFN_v3_PredictTime': [],
        'TabPFN_v3_Brier': [], 'TabPFN_v3_CalSlope': [], 'TabPFN_v3_CalIntercept': [], 'TabPFN_v3_Probas': [],
        'TabPFN_v35_MSE': [], 'TabPFN_v35_AUC': [], 'TabPFN_v35_Time': [], 'TabPFN_v35_FPR': [], 'TabPFN_v35_TPR': [],
        'TabPFN_v35_SetupTime': [], 'TabPFN_v35_TuneTime': [], 'TabPFN_v35_FitTime': [], 'TabPFN_v35_PredictTime': [],
        'TabPFN_v35_Brier': [], 'TabPFN_v35_CalSlope': [], 'TabPFN_v35_CalIntercept': [], 'TabPFN_v35_Probas': [],
        'EXAONE_MSE': [], 'EXAONE_AUC': [], 'EXAONE_Time': [], 'EXAONE_FPR': [], 'EXAONE_TPR': [],
        'EXAONE_SetupTime': [], 'EXAONE_TuneTime': [], 'EXAONE_FitTime': [], 'EXAONE_PredictTime': [],
        'EXAONE_Brier': [], 'EXAONE_CalSlope': [], 'EXAONE_CalIntercept': [], 'EXAONE_Probas': [],
        'TabICL-V2_MSE': [], 'TabICL-V2_AUC': [], 'TabICL-V2_Time': [], 'TabICL-V2_FPR': [], 'TabICL-V2_TPR': [],
        'TabICL-V2_SetupTime': [], 'TabICL-V2_TuneTime': [], 'TabICL-V2_FitTime': [], 'TabICL-V2_PredictTime': [],
        'TabICL-V2_Brier': [], 'TabICL-V2_CalSlope': [], 'TabICL-V2_CalIntercept': [], 'TabICL-V2_Probas': [],
        'y_test_all': []
    }

    if not include_exaone:
        errors = {key: values for key, values in errors.items() if not key.startswith("EXAONE_")}
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    if warnings_file is None:
        warnings_file = Path(config.REAL_RESULTS_DIR) / "warnings" / f"{time.time_ns()}.warnings.json"
    worker_dir = Path(warnings_file).with_suffix(".workers")
    with capture_output(warnings_file, config.REAL_RESULTS_DIR) as (captured_warnings, progress), parallel_config(
        backend="loky", inner_max_num_threads=1, initializer=initialize_worker,
    ):
        results = Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")(
            delayed(run_logged_iteration)(
                _single_real_iteration, worker_dir / f"iteration_{i + 1}.warnings.json",
                i, X_np, y_np, idx0, idx1,
                n1_tr, n2_tr, n1_te, n2_te, g_seed, include_exaone,
            )
            for i in range(iter)
        )

        for res, records, log_path in tqdm(results, total=iter, desc=progress_label or "Real-data resampling", file=progress, dynamic_ncols=True):
            captured_warnings.extend(records)
            print(Path(log_path).read_text(), end="")
            errors['y_test_all'].append(res['y_test'])
            for key, prefix in [
                ('L-SLR', 'L-SLR_'),
                ('SLR', 'SLR_'),
                ('RandomForest', 'RandomForest_'),
                ('XGBoost', 'XGBoost_'),
                ('CatBoost', 'CatBoost_'),
                ('Quadro', 'Quadro_'),
                ('TabPFN_v25', 'TabPFN_v25_'),
                ('TabPFN_v26', 'TabPFN_v26_'),
                ('TabPFN_v3', 'TabPFN_v3_'),
                ('TabPFN_v35', 'TabPFN_v35_'),
                ('EXAONE', 'EXAONE_'),
                ('TabICL-V2', 'TabICL-V2_')
            ]:
                if key == "EXAONE" and not include_exaone:
                    continue
                (t, mse, auc, fpr, tpr, proba, brier, slope, intercept,
                 setup_time, tune_time, fit_time, predict_time) = res[key]
                errors[prefix + 'Time'].append(t)
                errors[prefix + 'SetupTime'].append(setup_time)
                errors[prefix + 'TuneTime'].append(tune_time)
                errors[prefix + 'FitTime'].append(fit_time)
                errors[prefix + 'PredictTime'].append(predict_time)
                errors[prefix + 'MSE'].append(mse)
                errors[prefix + 'AUC'].append(auc)
                errors[prefix + 'FPR'].append(fpr)
                errors[prefix + 'TPR'].append(tpr)
                errors[prefix + 'Brier'].append(brier)
                errors[prefix + 'CalSlope'].append(slope)
                errors[prefix + 'CalIntercept'].append(intercept)
                errors[prefix + 'Probas'].append(proba)
    return errors



