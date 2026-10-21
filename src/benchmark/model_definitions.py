"""Model runners always return metrics followed by four times in seconds.

Return order:
    balanced_accuracy, auc, mse, fpr, tpr, proba, brier,
    setup_time, tune_time, fit_time, predict_time
"""
import os
from time import perf_counter

# ── BEFORE importing huggingface_hub ──
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"      # kills the "reconstructing file" bars
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np


from sklearn.metrics import roc_curve, balanced_accuracy_score, roc_auc_score, mean_squared_error, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
def _configure_pretrained_credentials():
    """Load literal tokens from .secrets/token.py; environment values win."""
    import ast
    from pathlib import Path

    names = {"HF_TOKEN", "TABPFN_TOKEN"}
    if all(os.environ.get(name) for name in names):
        return

    token_path = Path(__file__).resolve().parents[2] / ".secrets" / "token.py"
    if not token_path.is_file():
        return

    try:
        tree = ast.parse(token_path.read_text(encoding="utf-8"))
    except SyntaxError:
        raise ValueError("Invalid Python syntax in .secrets/token.py") from None

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            if os.environ.get(target.id):
                continue
            value = statement.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                raise ValueError(
                    f"{target.id} in .secrets/token.py must be a literal string"
                )
            if value.value:
                os.environ[target.id] = value.value


# Linear Sparse Logistic Regression
def run_linear_sparse_logistic_regression(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):
    """Scale explicitly, tune L1 logistic regression, fit once, and return metrics/times."""
    #setup
    setup_start = perf_counter()
    lr = LogisticRegression(
        penalty='l1', solver='liblinear', max_iter=2000,
        n_jobs=1, random_state=random_state,
    )
    param_grid = {'C': np.logspace(-4, 4, 10)}
    scoring = 'roc_auc'
    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=[(
            np.arange(len(X_train_sub)),
            np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)),
        )],
        scoring=scoring,
        n_jobs=1,
        refit=False,  # to time the single final fit
        error_score='raise',
    )
    setup_time = perf_counter() - setup_start

    # Tune
    tune_start = perf_counter()
    tune_scaler = StandardScaler()
    X_train_sub_scaled = tune_scaler.fit_transform(X_train_sub)
    X_val_scaled = tune_scaler.transform(X_val)
    grid_search.fit(
        np.vstack((X_train_sub_scaled, X_val_scaled)),
        np.hstack((y_train_sub, y_val)),
    )
    best_C = grid_search.best_params_['C']
    tune_time = perf_counter() - tune_start

    # fit
    fit_start = perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr.set_params(C=best_C)
    lr.fit(X_train_scaled, y_train)
    fit_time = perf_counter() - fit_start

    # predict
    predict_start = perf_counter()
    X_test_scaled = scaler.transform(X_test)
    proba = lr.predict_proba(X_test_scaled)[:, 1]
    pred = lr.classes_[(proba > 0.5).astype(int)]
    predict_time = perf_counter() - predict_start

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr, tpr, proba, brier,
        setup_time, tune_time, fit_time, predict_time,
    )


################################################################################


def run_sparse_logistic_regression(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):
    """Scale explicitly, tune L1 logistic regression, fit once, and return metrics/times."""
    
    #
    setup_start = perf_counter()
    def _augment_features(X):
        # Original features, squared terms, and pairwise products.
        rows, columns = np.triu_indices(X.shape[1])
        return np.hstack((X, X[:, rows] * X[:, columns]))

    lr = LogisticRegression(
        penalty='l1', solver='liblinear', max_iter=2000,
        n_jobs=1, random_state=random_state,
    )
    param_grid = {'C': np.logspace(-4, 4, 10)}
    scoring = 'roc_auc'
    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=[(
            np.arange(len(X_train_sub)),
            np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)),
        )],
        scoring=scoring,
        n_jobs=1,
        refit=False,  # for final fit below.
        error_score='raise',
    )
    setup_time = perf_counter() - setup_start

    #
    tune_start = perf_counter()
    Aug_X_train_sub = _augment_features(X_train_sub)
    Aug_X_val = _augment_features(X_val)
    tune_scaler = StandardScaler()
    X_train_sub_scaled = tune_scaler.fit_transform(Aug_X_train_sub)
    X_val_scaled = tune_scaler.transform(Aug_X_val)
    grid_search.fit(
        np.vstack((X_train_sub_scaled, X_val_scaled)),
        np.hstack((y_train_sub, y_val)),
    )
    best_C = grid_search.best_params_['C']
    tune_time = perf_counter() - tune_start

    #
    fit_start = perf_counter()
    Aug_X_train = _augment_features(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Aug_X_train)
    lr.set_params(C=best_C)
    lr.fit(X_train_scaled, y_train)
    fit_time = perf_counter() - fit_start

    # 
    predict_start = perf_counter()
    Aug_X_test = _augment_features(X_test)
    X_test_scaled = scaler.transform(Aug_X_test)
    proba = lr.predict_proba(X_test_scaled)[:, 1]
    pred = lr.classes_[(proba > 0.5).astype(int)]
    predict_time = perf_counter() - predict_start

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr, tpr, proba, brier,
        setup_time, tune_time, fit_time, predict_time,
    )


################################################################################


def run_random_forest(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):

    # 
    setup_start = perf_counter()
    rf = RandomForestClassifier(
        n_estimators=100, # number of trees
        max_depth=5, #important for overfiitng
        n_jobs=1,
        random_state=random_state
    ) 
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, None]}
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=[(np.arange(len(X_train_sub)), np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)))],
        scoring='roc_auc',
        n_jobs=1,
        refit=False,
    )
    setup_time = perf_counter() - setup_start

    # 
    tune_start = perf_counter()
    grid_search.fit(np.vstack((X_train_sub, X_val)), np.hstack((y_train_sub, y_val)))
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_depth = grid_search.best_params_['max_depth']
    tune_time = perf_counter() - tune_start

    # 
    fit_start = perf_counter()
    rf.set_params(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
    )
    rf.fit(X_train, y_train)
    fit_time = perf_counter() - fit_start

    # 
    predict_start = perf_counter()
    probabilities = rf.predict_proba(X_test)
    pred = rf.classes_[np.argmax(probabilities, axis=1)]
    proba = probabilities[:, 1]
    predict_time = perf_counter() - predict_start

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr,
        tpr,
        proba,
        brier,
        setup_time, tune_time, fit_time, predict_time,
    )


################################################################################


# XGBoost
def run_xgboost(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):
    """Choose tree count and depth using validation AUC, refit XGBoost, and evaluate test data."""
    
    #
    setup_start = perf_counter()
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        tree_method='hist',
        device='cpu',
        n_jobs=1,
        random_state=random_state
    ) 
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 6]}
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=[(np.arange(len(X_train_sub)), np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)))],
        scoring='roc_auc',
        n_jobs=1,
        refit=False,
    )
    setup_time = perf_counter() - setup_start

    # 
    tune_start = perf_counter()
    grid_search.fit(np.vstack((X_train_sub, X_val)), np.hstack((y_train_sub, y_val)))
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_depth = grid_search.best_params_['max_depth']
    tune_time = perf_counter() - tune_start

    # 
    fit_start = perf_counter()
    xgb.set_params(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
    )
    xgb.fit(X_train, y_train)
    fit_time = perf_counter() - fit_start

    # 
    predict_start = perf_counter()
    proba = xgb.predict_proba(X_test)[:, 1]
    pred = xgb.classes_[(proba > 0.5).astype(int)]
    predict_time = perf_counter() - predict_start

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr,
        tpr,
        proba,
        brier,
        setup_time, tune_time, fit_time, predict_time,
    )


################################################################################


# CatBoost
def run_catboost(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):
    """Choose iteration count and depth using validation AUC, refit CatBoost, and evaluate test data."""
    
    #
    setup_start = perf_counter()
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        thread_count=1, # training threads
        task_type='CPU',
        verbose=0,
        random_seed=random_state
    )
    def validation_auc(estimator, X, y):
        """CatBoost scoring must also request one prediction thread."""
        return roc_auc_score(y, estimator.predict_proba(X, thread_count=1)[:, 1])

    param_grid = {'iterations': [50, 100, 200], 'depth': [3, 5, 6]}
    grid_search = GridSearchCV(
        cb,
        param_grid,
        cv=[(np.arange(len(X_train_sub)), np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)))],
        scoring=validation_auc,
        n_jobs=1,
        refit=False,
    )
    setup_time = perf_counter() - setup_start

    # 
    tune_start = perf_counter()
    grid_search.fit(np.vstack((X_train_sub, X_val)), np.hstack((y_train_sub, y_val)))
    # Read the settings selected by the validation search.
    best_iterations = grid_search.best_params_['iterations']
    best_depth = grid_search.best_params_['depth']
    tune_time = perf_counter() - tune_start

    # 
    fit_start = perf_counter()
    cb.set_params(
        iterations=best_iterations,
        depth=best_depth,
    )
    cb.fit(X_train, y_train)
    fit_time = perf_counter() - fit_start

    # 
    predict_start = perf_counter()
    proba = cb.predict_proba(X_test, thread_count=1)[:, 1]
    pred = cb.classes_[(proba > cb.get_probability_threshold()).astype(int)]
    predict_time = perf_counter() - predict_start

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr,
        tpr,
        proba,
        brier,
        setup_time, tune_time, fit_time, predict_time,
    )


################################################################################
# Quadro
def run_quadro(X_train, X_test, y_train, y_test,
               X_train_sub, y_train_sub, X_val, y_val,
               random_state=None, *, metrics=None, solver="admm"):

    # 
    setup_start = perf_counter()
    from quadro import QUADROClassifier
    from threadpoolctl import threadpool_limits
    quadro = QUADROClassifier(standardize=False, solver=solver)
    penalty_pairs = [(0., 0.), (.01, .01), (.01, .1), (.1, .1), (.1, 1.)]
    param_grid = [
        {'lambda_quadratic': [quadratic], 'lambda_linear': [linear]}
        for quadratic, linear in penalty_pairs
    ]
    scoring = make_scorer(roc_auc_score, response_method="predict_proba")
    grid_search = GridSearchCV(
        quadro,
        param_grid,
        cv=[(
            np.arange(len(X_train_sub)),
            np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)),
        )],
        scoring=scoring,
        n_jobs=1,
        refit=False,  # same reason
        error_score='raise',
    )
    setup_time = perf_counter() - setup_start

    with threadpool_limits(limits=1):
        # 
        tune_start = perf_counter()
        tune_scaler = StandardScaler()
        X_train_sub_scaled = tune_scaler.fit_transform(X_train_sub)
        X_val_scaled = tune_scaler.transform(X_val)
        grid_search.fit(
            np.vstack((X_train_sub_scaled, X_val_scaled)),
            np.hstack((y_train_sub, y_val)),
        )
        best_lambda_quadratic = grid_search.best_params_['lambda_quadratic']
        best_lambda_linear = grid_search.best_params_['lambda_linear']
        tune_time = perf_counter() - tune_start

        # 
        fit_start = perf_counter()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        quadro.set_params(
            lambda_quadratic=best_lambda_quadratic,
            lambda_linear=best_lambda_linear,
        )
        quadro.fit(X_train_scaled, y_train)
        fit_time = perf_counter() - fit_start

        # 
        predict_start = perf_counter()
        X_test_scaled = scaler.transform(X_test)
        # Keep QUADRO's analytic class cutoff and calibrated probabilities.
        pred = quadro.predict(X_test_scaled)
        proba = quadro.predict_proba(X_test_scaled)[:, 1]
        predict_time = perf_counter() - predict_start

        fpr, tpr, _ = roc_curve(y_test, proba)
        brier = np.mean((proba - y_test) ** 2)
        return (
            balanced_accuracy_score(y_test, pred),
            roc_auc_score(y_test, proba),
            mean_squared_error(y_test, pred),
            fpr, tpr, proba, brier,
            setup_time, tune_time, fit_time, predict_time,
        )


################################################################################


def _run_tabpfn_generic(X_train, X_test, y_train, y_test, model_class, model_kwargs, random_state, *, setup_time=0):
    #step1 - setup - extra step here
    setup_start = perf_counter()
    model = model_class(**model_kwargs,
                        ignore_pretraining_limits=True,
                        random_state=random_state
                        )
    setup_time += perf_counter() - setup_start

    # step2 - No hyperparameter 
    tune_time = 0.0  

    # step3 - fit
    fit_start = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - fit_start

    # step4 - predict
    predict_start = perf_counter()
    probabilities = model.predict_proba(X_test)
    pred = model.classes_[np.argmax(probabilities, axis=1)]
    proba = probabilities[:, 1]
    predict_time = perf_counter() - predict_start

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr, tpr, proba, brier,
        setup_time, tune_time, fit_time, predict_time,
    )


# Runner for each version
def run_tabpfn_v25(X_train, X_test, y_train, y_test, random_state=None):
    #
    setup_start = perf_counter()
    _configure_pretrained_credentials()
    from tabpfn_v25 import TabPFNClassifier as TabPFNClassifier_v25
    setup_time = perf_counter() - setup_start
    
    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v25,
        {"n_estimators": 8, "device": "cpu", "n_preprocessing_jobs": 1}, #e_est =8 default
        random_state, setup_time=setup_time,
    )

def run_tabpfn_v26(X_train, X_test, y_train, y_test, random_state=None):
    #
    setup_start = perf_counter()
    _configure_pretrained_credentials()
    from tabpfn_v26 import TabPFNClassifier as TabPFNClassifier_v26
    setup_time = perf_counter() - setup_start
    
    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v26,
         {"n_estimators": 8, "device": "cpu", "n_preprocessing_jobs": 1}, #e_est =8 default
        random_state, setup_time=setup_time,
    )

def run_tabpfn_v3(X_train, X_test, y_train, y_test, random_state=None):
    #
    setup_start = perf_counter()
    _configure_pretrained_credentials()
    from tabpfn_v3 import TabPFNClassifier as TabPFNClassifier_v3
    setup_time = perf_counter() - setup_start
    #
    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v3,
        {"n_estimators": "auto", "device": "cpu", "n_preprocessing_jobs": 1}, #e_est = 'auto' default
        random_state, setup_time=setup_time,
    )

def run_tabpfn_v35(X_train, X_test, y_train, y_test, random_state=None):
    #
    setup_start = perf_counter()
    _configure_pretrained_credentials()
    from tabpfn_v35 import TabPFNClassifier as TabPFNClassifier_v35
    setup_time = perf_counter() - setup_start
    #
    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v35,
        {"n_estimators": "auto", "device": "cpu", "n_preprocessing_jobs": 1}, #e_est = 'auto' default
        random_state, setup_time=setup_time,
    )
    


################################################################################
# EXAONE
def run_exaone(X_train, X_test, y_train, y_test, random_state=None):
    #
    setup_start = perf_counter()
    _configure_pretrained_credentials()
    from exaonetabular import EXAONETabularClassifier
    from .execution import single_threaded_torch
    with single_threaded_torch(n_threads=1):
        clf = EXAONETabularClassifier.from_pretrained(
            device="cpu",
            ensemble_count=8, #e_ensem =8 default
            seed=random_state,
        )
        setup_time = perf_counter() - setup_start
        
        # foundation model
        tune_time = 0.0  
        
        #
        fit_start = perf_counter()
        clf.fit(X_train, y_train)
        fit_time = perf_counter() - fit_start
        
        #
        predict_start = perf_counter()
        probabilities = clf.predict_proba(X_test)
        pred = clf.classes_[np.argmax(probabilities, axis=1)]
        proba = probabilities[:, 1]
        predict_time = perf_counter() - predict_start

        fpr, tpr, _ = roc_curve(y_test, proba)
        brier = np.mean((proba - y_test) ** 2)
        return (
            balanced_accuracy_score(y_test, pred),
            roc_auc_score(y_test, proba),
            mean_squared_error(y_test, pred),
            fpr, tpr, proba, brier,
            setup_time, tune_time, fit_time, predict_time,
        )
    
    
    ################################################################################
# TabICLv2
def run_tabicl(X_train, X_test, y_train, y_test, random_state=None):
    #
    setup_start = perf_counter()
    _configure_pretrained_credentials()
    from tabicl import TabICLClassifier
    clf = TabICLClassifier(
        device="cpu",
        n_jobs=1,
        n_estimators=8,  #e_est=8 default
        verbose=False,
        random_state=random_state if random_state is not None else 42,
    )
    setup_time = perf_counter() - setup_start
    
    # 
    tune_time = 0.0  
    
    #
    fit_start = perf_counter()
    clf.fit(X_train, y_train)
    fit_time = perf_counter() - fit_start
    
    #
    predict_start = perf_counter()
    probabilities = clf.predict_proba(X_test)
    pred = clf.classes_[np.argmax(probabilities, axis=1)]
    proba = probabilities[:, 1]
    predict_time = perf_counter() - predict_start

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr, tpr, proba, brier,
        setup_time, tune_time, fit_time, predict_time,
    )




