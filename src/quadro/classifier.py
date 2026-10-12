"""Scikit-learn-compatible QUADRO projection and binary classifier."""

import numpy as np
from scipy.special import ndtr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets

from .moments import estimate_moments, estimate_kurtosis, projected_moments
from .optimization import solve_quadro


def classification_threshold(means, variances, priors):
    """Globally minimize the normal-approximation threshold risk (equations 17–18).

    The stationary points solve prior0*pdf0(c)=prior1*pdf1(c). Compare all
    real roots and finite approximations of both constant-decision limits.
    """
    m0, m1 = means
    v0, v1 = variances
    coefficients = [
        0.5 / v1 - 0.5 / v0,
        m0 / v0 - m1 / v1,
        m1 * m1 / (2 * v1)
        - m0 * m0 / (2 * v0)
        + np.log(priors[0] / priors[1])
        + 0.5 * np.log(v1 / v0),
    ]
    if abs(coefficients[0]) <= 1e-12 * max(1, abs(coefficients[1])):
        roots = (
            [] if abs(coefficients[1]) < 1e-12 else [-coefficients[2] / coefficients[1]]
        )
    else:
        roots = [float(r.real) for r in np.roots(coefficients) if abs(r.imag) < 1e-10]
    width = 12 * np.sqrt(max(variances))
    candidates = roots + [min(means) - width, max(means) + width, (m0 + m1) / 2]

    def risk(c):
        return priors[0] * ndtr((c - m0) / np.sqrt(v0)) + priors[1] * ndtr(
            (m1 - c) / np.sqrt(v1)
        )

    return float(min(candidates, key=risk))


class QUADROClassifier(ClassifierMixin, BaseEstimator):
    """Sparse quadratic Rayleigh optimization based on Fan et al. (2015).

    Implements elliptical projected moments and the constrained L1 objective.
    `solver='admm'` uses matrix-free ADMM; `solver='linearized_alm'` uses
    author-style alternating linearized augmented-Lagrangian updates.
    Robust covariance uses Catoni means/variances, Kendall tau-a,
    and the max-norm PSD projection. Automatic kurtosis uses CLIME precision
    estimation; a regularized covariance inverse is also selectable. `gamma=0` is the Gaussian model; gamma=2/(df-4)
    is appropriate for a covariance-normalized t distribution with df>4.

    `predict` follows the paper's analytic threshold. `predict_proba` adds
    Platt calibration fitted only on training projections; its 0.5 cutoff can
    differ from the paper's classification threshold.
    """

    def __init__(
        self,
        lambda_quadratic=0.01,
        lambda_linear=0.01,
        *,
        robust=True,
        gamma="auto",
        kappa=None,
        ridge=1e-4,
        precision="clime",
        clime_lambda=0.1,
        rho=1.0,
        max_iter=2000,
        tol=1e-5,
        standardize=True,
        solver="admm",
    ):
        self.lambda_quadratic = lambda_quadratic
        self.lambda_linear = lambda_linear
        self.robust = robust
        self.gamma = gamma
        self.kappa = kappa
        self.ridge = ridge
        self.precision = precision
        self.clime_lambda = clime_lambda
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.solver = solver

    def fit(self, X, y):
        if self.solver not in ("admm", "linearized_alm"):
            raise ValueError("solver must be 'admm' or 'linearized_alm'")
        if self.lambda_quadratic < 0 or self.lambda_linear < 0:
            raise ValueError("QUADRO penalties must be nonnegative")
        if self.ridge <= 0 or self.rho <= 0 or self.tol <= 0 or self.max_iter < 1:
            raise ValueError("ridge, rho, tol, and max_iter must be positive")
        if self.gamma != "auto" and (
            not np.isscalar(self.gamma) or not np.isfinite(self.gamma) or self.gamma < 0
        ):
            raise ValueError("gamma must be 'auto' or a finite nonnegative number")
        if self.kappa is not None and (not np.isfinite(self.kappa) or self.kappa <= 0):
            raise ValueError("kappa must be positive")
        if self.precision not in ("clime", "inverse"):
            raise ValueError("precision must be 'clime' or 'inverse'")
        if not np.isfinite(self.clime_lambda) or not 0 <= self.clime_lambda < 1:
            raise ValueError("clime_lambda must be in [0, 1)")
        X, y = validate_data(self, X, y, dtype=float)
        check_classification_targets(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(self.classes_) != 2 or np.min(counts) < 2:
            raise ValueError(
                "QUADRO requires exactly two classes with at least two observations each"
            )
        self.scaler_ = StandardScaler(
            with_mean=self.standardize, with_std=self.standardize
        ).fit(X)
        transformed = self.scaler_.transform(X)
        groups = [transformed[y == label] for label in self.classes_]
        moments = [
            estimate_moments(group, robust=self.robust, ridge=self.ridge)
            for group in groups
        ]
        self.means_ = np.array([item[0] for item in moments])
        self.covariances_ = np.array([item[1] for item in moments])
        self.class_prior_ = counts / counts.sum()
        self.kappa_ = counts[1] / counts[0] if self.kappa is None else self.kappa
        self.gamma_ = (
            float(
                np.average(
                    [
                        estimate_kurtosis(
                            group, *moment, self.precision, self.clime_lambda
                        )
                        for group, moment in zip(groups, moments)
                    ],
                    weights=counts,
                )
            )
            if self.gamma == "auto"
            else float(self.gamma)
        )
        optimizer = solve_quadro
        if self.solver == "linearized_alm":
            from .linearized import solve_quadro_alm
            optimizer = solve_quadro_alm
        self.omega_, self.delta_, self.n_iter_, self.residual_, self.converged_ = (
            optimizer(
                self.means_,
                self.covariances_,
                self.gamma_,
                self.kappa_,
                self.lambda_quadratic,
                self.lambda_linear,
                rho=self.rho,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        )
        projected = [
            projected_moments(self.omega_, self.delta_, mean, covariance, self.gamma_)
            for mean, covariance in zip(self.means_, self.covariances_)
        ]
        self.projected_means_ = np.array([m for m, _ in projected])
        self.projected_variances_ = np.array([v for _, v in projected])
        self.threshold_ = classification_threshold(
            self.projected_means_, self.projected_variances_, self.class_prior_
        )
        self.calibration_scaler_ = StandardScaler().fit(
            self.decision_function(X).reshape(-1, 1)
        )
        self.calibrator_ = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(
            self.calibration_scaler_.transform(
                self.decision_function(X).reshape(-1, 1)
            ),
            y,
        )
        return self

    def transform(self, X):
        """Return the learned one-dimensional quadratic projection Q(X)."""
        check_is_fitted(self, "omega_")
        X = self.scaler_.transform(validate_data(self, X, reset=False, dtype=float))
        return (
            np.einsum("ij,ij->i", X @ self.omega_, X) - 2 * X @ self.delta_
        ).reshape(-1, 1)

    def decision_function(self, X):
        check_is_fitted(self, "threshold_")
        return self.threshold_ - self.transform(X).ravel()

    def predict(self, X):
        check_is_fitted(self, "classes_")
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def predict_proba(self, X):
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.predict_proba(
            self.calibration_scaler_.transform(
                self.decision_function(X).reshape(-1, 1)
            ),
        )
