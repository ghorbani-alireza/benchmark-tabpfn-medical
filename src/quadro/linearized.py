"""Author-style linearized augmented Lagrangian, implemented independently.

Alternating soft-threshold block updates follow InnerLoop.m in the supplied
archive. Storage, iteration safeguards, and step-size bounds are adapted; this
is a second solver for the same estimator, not an oracle-tuned MATLAB replica.
"""

import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.exceptions import ConvergenceWarning

from .optimization import smooth_gradient


def solve_quadro_alm(
    means,
    covariances,
    gamma,
    kappa,
    lambda_quadratic,
    lambda_linear,
    *,
    rho=1.0,
    max_iter=2000,
    tol=1e-5,
    max_inner_iter=2000,
):
    """Alternative solver: update quadratic and linear weights in alternating steps.

    Solve the unit-contrast L1 problem with a linearized augmented Lagrangian.

    Use true block Hessian bounds (including 8*Sigma in the delta block),
    matrix-free eigenvalue estimation, bounded inner/outer loops, and the
    same contrast orientation as ADMM. Return the same convergence diagnostics.
    """
    d = len(means[0])
    contrast_q = (
        covariances[0]
        + np.outer(means[0], means[0])
        - covariances[1]
        - np.outer(means[1], means[1])
    )
    contrast_l = -2 * (means[0] - means[1])
    if np.linalg.norm(contrast_q) + np.linalg.norm(contrast_l) < 1e-12:
        raise ValueError("QUADRO cannot separate identical estimated class moments")

    def contrast(omega, delta):
        return np.sum(contrast_q * omega) + contrast_l @ delta

    def hessian_q(vector):
        matrix = vector.reshape(d, d)
        matrix = (matrix + matrix.T) / 2
        gradient, _ = smooth_gradient(
            means, covariances, gamma, kappa, matrix, np.zeros(d)
        )
        return (gradient + rho * np.sum(contrast_q * matrix) * contrast_q).ravel()

    if d == 1:
        tau_q = hessian_q(np.ones(1))[0]
    else:
        operator = LinearOperator((d * d, d * d), matvec=hessian_q)
        initial = np.random.default_rng(0).normal(size=d * d)
        tau_q = eigsh(
            operator, k=1, which="LA", v0=initial, tol=1e-9, return_eigenvectors=False
        )[0]
    hessian_l = 8 * (covariances[0] + kappa * covariances[1]) + rho * np.outer(
        contrast_l, contrast_l
    )
    # Small safety margin for finite precision in the eigenvalue estimates.
    tau_q = max(float(tau_q) * 1.01, np.finfo(float).eps)
    tau_l = max(float(np.linalg.eigvalsh(hessian_l)[-1]) * 1.01, np.finfo(float).eps)

    def shrink(value, level):
        return np.sign(value) * np.maximum(np.abs(value) - level, 0)

    omega, delta, multiplier = np.zeros((d, d)), np.zeros(d), 0.0
    converged = False
    residual = np.inf
    for iteration in range(1, max_iter + 1):
        previous_q, previous_l = omega.copy(), delta.copy()
        inner_residual = np.inf
        for _ in range(max_inner_iter):
            gradient_q, _ = smooth_gradient(
                means, covariances, gamma, kappa, omega, delta
            )
            gradient_q += (multiplier + rho * (contrast(omega, delta) - 1)) * contrast_q
            updated_q = shrink(omega - gradient_q / tau_q, lambda_quadratic / tau_q)
            updated_q = (updated_q + updated_q.T) / 2
            _, gradient_l = smooth_gradient(
                means, covariances, gamma, kappa, updated_q, delta
            )
            gradient_l += (
                multiplier + rho * (contrast(updated_q, delta) - 1)
            ) * contrast_l
            updated_l = shrink(delta - gradient_l / tau_l, lambda_linear / tau_l)
            inner_residual = max(
                tau_q * np.linalg.norm(updated_q - omega),
                tau_l * np.linalg.norm(updated_l - delta),
            )
            omega, delta = updated_q, updated_l
            if inner_residual <= tol / 4:
                break
        feasibility = contrast(omega, delta) - 1
        multiplier += rho * feasibility
        residual = max(
            abs(feasibility),
            np.linalg.norm(omega - previous_q),
            np.linalg.norm(delta - previous_l),
            inner_residual,
        )
        if residual <= tol:
            converged = True
            break
    if not converged:
        warnings.warn(
            "QUADRO ALM optimizer did not converge; increase iteration limits or adjust rho",
            ConvergenceWarning,
        )
    normalization = contrast(omega, delta)
    if abs(normalization) < 1e-12:
        raise RuntimeError("QUADRO ALM optimizer produced an infeasible zero direction")
    return omega / normalization, delta / normalization, iteration, residual, converged
