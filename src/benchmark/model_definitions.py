import os

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
    # model and tuning grid
    """Tune L1 logistic regression on the validation split, refit on all training data, and evaluate test data."""
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
    """Add squared features and pairwise products, then tune and evaluate L1 logistic regression."""
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
    """Choose tree count and depth using validation AUC, refit the forest, and evaluate test data."""
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
    """Choose tree count and depth using validation AUC, refit XGBoost, and evaluate test data."""
    from xgboost import XGBClassifier

    # Define model and tuning grid
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        tree_method='hist',
        device='cpu',
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
    """Choose iteration count and depth using validation AUC, refit CatBoost, and evaluate test data."""
    from catboost import CatBoostClassifier
    #  model and tuning grid
    cb = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        thread_count=1, # training threads
        task_type='CPU',
        verbose=0,
        random_seed=random_state
    )# the same as XGBoost as much as possible
    def validation_auc(estimator, X, y):
        """CatBoost scoring must also request one prediction thread."""
        return roc_auc_score(y, estimator.predict_proba(X, thread_count=1)[:, 1])

    param_grid = {'iterations': [50, 100, 200], 'depth': [3, 5, 6]}
    grid_search = GridSearchCV(
        cb,
        param_grid,
        cv=[(np.arange(len(X_train_sub)), np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)))],
        scoring=validation_auc,
        n_jobs=1
    )
    grid_search.fit(np.vstack((X_train_sub, X_val)), np.hstack((y_train_sub, y_val)))
    best_model = grid_search.best_estimator_

    # train best model
    best_model.fit(X_train, y_train)

    # predict/ compute metrics
    pred = best_model.predict(X_test, thread_count=1)
    proba = best_model.predict_proba(X_test, thread_count=1)[:, 1]
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
# Quadro
def run_quadro(X_train, X_test, y_train, y_test,
               X_train_sub, y_train_sub, X_val, y_val,
               random_state=None, *, metrics=None, solver="admm"):
    """Choose QUADRO penalties with GridSearchCV, then evaluate on test data.

    random_state is accepted like the other runners; QUADRO's fit is deterministic.
    """
    from quadro import QUADROClassifier
    from threadpoolctl import threadpool_limits

    # five  penalty pairs, rather than every combination.
    penalty_pairs = [(0.0, 0.0), (0.01, 0.01), (0.01, 0.1), (0.1, 0.1), (0.1, 1.0)]
    param_grid = [
        {'lambda_quadratic': [quadratic], 'lambda_linear': [linear]}
        for quadratic, linear in penalty_pairs
    ]
    # The same as other classical runners.
    validation_split = [(
        np.arange(len(X_train_sub)),
        np.arange(len(X_train_sub), len(X_train_sub) + len(X_val)),
    )]
    # Preserve the previous probability-based AUC criterion. The default
    # 'roc_auc' scorer would prefer QUADRO's uncalibrated decision_function.
    scoring = make_scorer(roc_auc_score, response_method="predict_proba")

    with threadpool_limits(limits=1):
        grid_search = GridSearchCV(
            QUADROClassifier(solver=solver),
            param_grid,
            cv=validation_split,
            scoring=scoring,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        grid_search.fit(
            np.vstack((X_train_sub, X_val)),
            np.hstack((y_train_sub, y_val)),
        )
        # GridSearchCV already refits the winner on all supplied training rows.
        model = grid_search.best_estimator_

        # Preserve QUADRO's analytic class cutoff and calibrated probabilities.
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        brier = np.mean((proba - y_test) ** 2)
        return (
            balanced_accuracy_score(y_test, pred),
            roc_auc_score(y_test, proba),
            mean_squared_error(y_test, pred),
            fpr, tpr, proba, brier,
        )


################################################################################

# TabPFN
# Generic runner to use for all TabPFN versions  
def _run_tabpfn_generic(X_train, X_test, y_train, y_test, model_class, model_kwargs, random_state):
    
    # scale features
    """Scale features using training data, fit the selected pretrained TabPFN version, and return test measures."""
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
    """Evaluate the vendored TabPFN 2.5 classifier on CPU with the supplied random seed."""
    _configure_pretrained_credentials()
    from tabpfn_v25 import TabPFNClassifier as TabPFNClassifier_v25

    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v25,
        {"n_estimators": 1, "device": "cpu", "n_preprocessing_jobs": 1},
        random_state
    )

def run_tabpfn_v26(X_train, X_test, y_train, y_test, random_state=None):
    """Evaluate the vendored TabPFN 2.6 classifier on CPU with the supplied random seed."""
    _configure_pretrained_credentials()
    from tabpfn_v26 import TabPFNClassifier as TabPFNClassifier_v26

    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v26,
         {"n_estimators": 1, "device": "cpu", "n_preprocessing_jobs": 1},
        random_state
    )

def run_tabpfn_v3(X_train, X_test, y_train, y_test, random_state=None):
    """Evaluate the vendored TabPFN 3 classifier on CPU with the supplied random seed."""
    _configure_pretrained_credentials()
    from tabpfn_v3 import TabPFNClassifier as TabPFNClassifier_v3

    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v3,
        {"n_estimators": 1, "device": "cpu", "n_preprocessing_jobs": 1},
        random_state
    )

def run_tabpfn_v35(X_train, X_test, y_train, y_test, random_state=None):
    """Evaluate the vendored TabPFN 3.5 classifier on CPU with the supplied random seed."""
    _configure_pretrained_credentials()
    from tabpfn_v35 import TabPFNClassifier as TabPFNClassifier_v35

    return _run_tabpfn_generic(
        X_train, X_test, y_train, y_test,
        TabPFNClassifier_v35,
        {"n_estimators": 1, "device": "cpu", "n_preprocessing_jobs": 1},
        random_state
    )
    


################################################################################
# EXAONE
def run_exaone(X_train, X_test, y_train, y_test, random_state=None):
    """Load pretrained EXAONE using the supplied seed, fit training data, and evaluate test data on CPU."""
    _configure_pretrained_credentials()
    from exaonetabular import EXAONETabularClassifier
    from .execution import single_threaded_torch

    with single_threaded_torch(n_threads=1):
        clf = EXAONETabularClassifier.from_pretrained(
            device="cpu",
            ensemble_count=1,
            seed=random_state,
        )
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
    
    
    ################################################################################
# TabICLv2
def run_tabicl(X_train, X_test, y_train, y_test, random_state=None):
    """Fit the pretrained TabICL classifier and evaluate test data on CPU without verbose messages."""
    _configure_pretrained_credentials()
    from tabicl import TabICLClassifier
    clf = TabICLClassifier(
        device="cpu",
        n_jobs=1,
        n_estimators=1,
        verbose=False,
        random_state=random_state if random_state is not None else 42,
    )
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




