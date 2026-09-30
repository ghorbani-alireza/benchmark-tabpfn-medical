import os, sys
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "vendor"))


import numpy as np
import pandas as pd
from huggingface_hub import login # tabpfn needs huggingface
from importlib.metadata import version 

from sklearn.metrics import roc_curve, balanced_accuracy_score, roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
print("scikit‑learn package version:", version("scikit-learn"))

from xgboost import XGBClassifier
print("XGBoost package version:", version("xgboost"))

from catboost import CatBoostClassifier
print("CatBoost package version:", version("catboost"))

# Force JAX to CPU and single-threaded — MUST come before tabfm/jax import
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# then, existing imports:
from autogluon.tabular import TabularPredictor
import autogluon.common
print("AutoGluon (Mitra) package version:", getattr(autogluon.common, "__version__", "unknown"))

import tabfm
from tabfm import TabFMClassifier
from tabfm.src.jax import tabfm_v1_0_0 as tabfm_v1_0_0
print("TabFM package version:", getattr(tabfm, "__version__", "unknown"))


# TabPFN - read tokens from  config.py
from .tokens import (
    HF_TOKEN,
    TABPFN_TOKEN
)


os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
login(token=HF_TOKEN)


try:
    from tabpfn_v25 import TabPFNClassifier as TabPFNClassifier_v25
    from tabpfn_v26 import TabPFNClassifier as TabPFNClassifier_v26
    from tabpfn_v3  import TabPFNClassifier as TabPFNClassifier_v3
    from tabpfn_v35 import TabPFNClassifier as TabPFNClassifier_v35
    _TABPFN_VERSIONS_READY = True
except ModuleNotFoundError:
    TabPFNClassifier_v25 = TabPFNClassifier_v26 = None
    TabPFNClassifier_v3  = TabPFNClassifier_v35 = None
    _TABPFN_VERSIONS_READY = False
    print("⚠️ tabpfn_v25/v26/v3/v35 not installed — run installer first.")



if _TABPFN_VERSIONS_READY:
    for _p in ("tabpfn_v25", "tabpfn_v26", "tabpfn_v3", "tabpfn_v35"):
        print(_p, "loaded")
            





# Linear Sparse Logistic Regression
def run_linear_sparse_logistic_regression(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):
    # model and tuning grid
    lr = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        max_iter=2000,
        n_jobs=1,
        random_state=random_state
    )
    param_grid = {'C': np.logspace(-4, 4, 10)}
    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=[(np.arange(len(X_train_sub)), np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)))],  # indices for train/val
        scoring='roc_auc',
        n_jobs=1
    )
    grid_search.fit(np.vstack((X_train_sub, X_val)), np.hstack((y_train_sub, y_val)))
    best_model = grid_search.best_estimator_
    # page 276 ITSL book descibes GridSearchCV()

    # best model on full training data
    best_model.fit(X_train, y_train)

    # predict + compute metrics
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr,
        tpr,
        proba,
        brier
    )



################################################################################



# Sparse Logistic Regression on Augmented Feature Space
def run_sparse_logistic_regression(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):

    # augment features Function
    def _augment_features(X):
        n, d = X.shape
        idx = np.triu_indices(d, k=0)  # indices for upper triangle + diagonal
        xx = X[:, idx[0]] * X[:, idx[1]]  # element-wise multiplication
        return np.hstack((X, xx))

    # augmentation first
    Aug_X_train = _augment_features(X_train)
    Aug_X_test = _augment_features(X_test)
    Aug_X_train_sub = _augment_features(X_train_sub)
    Aug_X_val = _augment_features(X_val)

    # model and tuning grid
    lr = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        max_iter=2000,
        n_jobs=1,
        random_state=random_state
    )
    param_grid = {'C': np.logspace(-4, 4, 10)}
    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=[(np.arange(len(Aug_X_train_sub)), np.arange(len(Aug_X_train_sub), len(Aug_X_train_sub) + len(Aug_X_val)))],
        scoring='roc_auc',
        n_jobs=1
    )
    grid_search.fit(np.vstack((Aug_X_train_sub, Aug_X_val)), np.hstack((y_train_sub, y_val)))
    best_model = grid_search.best_estimator_

    # best model
    best_model.fit(Aug_X_train, y_train)

    # predict
    pred = best_model.predict(Aug_X_test)
    proba = best_model.predict_proba(Aug_X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr,
        tpr,
        proba,
        brier
    )

################################################################################


# random forests
def run_random_forest(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):

    # model and tuning grid
    rf = RandomForestClassifier(
        n_estimators=100, # number of decision trees
        max_depth=5, #important for overfiitng
        n_jobs=1,
        random_state=random_state
    ) # consistent with XGBoost and CatBoost also
    # not include too many parameteres cosidering the grid length
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, None]}
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=[(np.arange(len(X_train_sub)), np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)))],
        scoring='roc_auc',
        n_jobs=1
    )
    grid_search.fit(np.vstack((X_train_sub, X_val)), np.hstack((y_train_sub, y_val)))
    best_model = grid_search.best_estimator_

    # train best model on full train
    best_model.fit(X_train, y_train)

    # predict and compute metrics
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr,
        tpr,
        proba,
        brier
    )


################################################################################


# XGBoost
def run_xgboost(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):

    # Define model and tuning grid
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=1,
        random_state=random_state
    ) # the same as random forests and catboost
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 6]}
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=[(np.arange(len(X_train_sub)), np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)))],
        scoring='roc_auc',
        n_jobs=1
    )
    grid_search.fit(np.vstack((X_train_sub, X_val)), np.hstack((y_train_sub, y_val)))
    best_model = grid_search.best_estimator_

    # best model on full train data
    best_model.fit(X_train, y_train)

    # predict and compute metrics
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr,
        tpr,
        proba,
        brier
    )


################################################################################


# CatBoost
def run_catboost(X_train, X_test, y_train, y_test, X_train_sub, y_train_sub, X_val, y_val, random_state=None):
    #  model and tuning grid
    cb = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        thread_count=1, # like n_job
        verbose=0,
        random_seed=random_state
    )# the same as possible
    param_grid = {'iterations': [50, 100, 200], 'depth': [3, 5, 6]}
    grid_search = GridSearchCV(
        cb,
        param_grid,
        cv=[(np.arange(len(X_train_sub)), np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)))],
        scoring='roc_auc',
        n_jobs=1
    )
    grid_search.fit(np.vstack((X_train_sub, X_val)), np.hstack((y_train_sub, y_val)))
    best_model = grid_search.best_estimator_

    # train best model
    best_model.fit(X_train, y_train)

    # predict/ compute metrics
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr,
        tpr,
        proba,
        brier
    )


################################################################################

# TabPFN
# Generic runner to use for all TabPFN versions  
def _run_tabpfn_generic(X_train, X_test, y_train, y_test, model_class, model_kwargs, random_state):
    
    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = model_class(**model_kwargs, 
                        ignore_pretraining_limits=True,
                        random_state=random_state
                        )
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr, tpr, proba, brier
    )
    

# Runner for each version
def run_tabpfn_v25(X_train, X_test, y_train, y_test, random_state=None):
    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v25,
        {"n_estimators": 1, "device": "cpu", "n_preprocessing_jobs": 1},
        random_state
    )

def run_tabpfn_v26(X_train, X_test, y_train, y_test, random_state=None):
    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v26,
         {"n_estimators": 1, "device": "cpu", "n_preprocessing_jobs": 1},
        random_state
    )

def run_tabpfn_v3(X_train, X_test, y_train, y_test, random_state=None):
    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v3,
        {"n_estimators": 1, "device": "cpu", "n_preprocessing_jobs": 1},
        random_state
    )

def run_tabpfn_v35(X_train, X_test, y_train, y_test, random_state=None):
    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v35,
        {"n_estimators": 1, "device": "cpu", "n_preprocessing_jobs": 1},
        random_state
    )
    


################################################################################
# Mitra-V2
def run_mitra_v2(X_train, X_test, y_train, y_test, random_state=None):
    # AutoGluon expects a DataFrame with the label column included
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train
    test_df = pd.DataFrame(X_test)

    predictor = TabularPredictor(
        label='target',
        verbosity=0,
        learner_kwargs={'random_state': random_state},   # ← only valid place
    )
    predictor.fit(
        train_df,
        hyperparameters={'MITRA': {'fine_tune': False}},
        num_cpus=1,        #  single CPU
        num_gpus=0         #  no GPU
    )

    pred = predictor.predict(test_df).values
    proba = predictor.predict_proba(test_df).iloc[:, 1].values

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr, tpr, proba, brier
    )       



################################################################################
# TabFM
def run_tabfm(X_train, X_test, y_train, y_test, random_state=None):
    # Load the pre-trained TabFM model for classification
    model = tabfm_v1_0_0.load(model_type="classification")
    clf = TabFMClassifier(model=model)#, random_state=random_state)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, proba)
    brier = np.mean((proba - y_test) ** 2)
    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        mean_squared_error(y_test, pred),
        fpr, tpr, proba, brier
    )