"""Elliptical moment estimates for QUADRO; no fourth-cross-moment expansion."""

import numpy as np
from scipy.stats import kendalltau


def catoni_mean(X):
    """Coordinatewise Catoni M-estimates using Section 4.1's influence function.

    The variance bound is three times the largest sample marginal variance.
    Confidence is 1/max(n,d)^2, capped for samples too small for its formula.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    variance_bound = 3 * np.max(np.var(X, axis=0, ddof=1))
    if variance_bound <= np.finfo(float).eps:
        return X.mean(axis=0)
    log_inverse_confidence = min(2 * np.log(max(n, d)), n / 4)
    alpha = np.sqrt(
        2
        * log_inverse_confidence
        * (n - 2 * log_inverse_confidence)
        / (n * n * variance_bound)
    )
    lower, upper = X.min(axis=0), X.max(axis=0)
    for _ in range(70):
        midpoint = (lower + upper) / 2
        residual = alpha * (X - midpoint)
        influence = np.sign(residual) * np.log1p(np.abs(residual) + residual**2 / 2)
        positive = influence.sum(axis=0) > 0
        lower = np.where(positive, midpoint, lower)
        upper = np.where(positive, upper, midpoint)
    return (lower + upper) / 2


def _nearest_psd_max_norm(matrix, tol=1e-7, max_iter=2000):
    """Solve min_{A PSD} ||A-matrix||_max by ADMM (paper equation 11).

    Moreau's identity gives the max-norm proximal map using an L1-ball projection.
    """
    if np.linalg.eigvalsh(matrix)[0] >= 0:
        return matrix.copy()
    primal, dual = matrix.copy(), np.zeros_like(matrix)
    rho = 1.0
    scale = max(1.0, np.linalg.norm(matrix))
    for _ in range(max_iter):
        eigenvalues, eigenvectors = np.linalg.eigh(primal - dual)
        psd = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        offset = psd + dual - matrix
        absolute = np.abs(offset).ravel()
        radius = 1 / rho
        if absolute.sum() <= radius:
            projection = offset
        else:
            ordered = np.sort(absolute)[::-1]
            cumulative = np.cumsum(ordered)
            indices = np.arange(1, len(ordered) + 1)
            active = np.flatnonzero(ordered > (cumulative - radius) / indices)[-1]
            level = (cumulative[active] - radius) / (active + 1)
            projection = np.sign(offset) * np.maximum(np.abs(offset) - level, 0)
        updated = matrix + offset - projection
        dual += psd - updated
        primal_residual = np.linalg.norm(psd - updated)
        dual_residual = rho * np.linalg.norm(updated - primal)
        residual = max(primal_residual, dual_residual)
        primal = updated
        if primal_residual > 10 * dual_residual:
            rho *= 2
            dual /= 2
        elif dual_residual > 10 * primal_residual:
            rho /= 2
            dual *= 2
        if residual <= tol * scale:
            # Return the PSD iterate, rather than the box-side iterate.
            return (psd + psd.T) / 2
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    warnings.warn(
        "QUADRO covariance PSD projection did not converge", ConvergenceWarning
    )
    return (psd + psd.T) / 2


def estimate_moments(X, robust=True, ridge=1e-4):
    """Estimate class mean and PSD covariance using Sections 4.1–4.2."""
    n, d = X.shape
    if not robust:
        mean = X.mean(axis=0)
        centered = X - mean
        covariance = centered.T @ centered / (n - 1)
    else:
        mean = catoni_mean(X)
        variance = np.maximum(catoni_mean(X**2) - mean**2, ridge)
        correlation = np.eye(d)
        pairs = n * (n - 1) / 2
        untied = []
        for j in range(d):
            _, counts = np.unique(X[:, j], return_counts=True)
            untied.append(pairs - np.sum(counts * (counts - 1) / 2))
        for j in range(d):
            for k in range(j):
                tau = kendalltau(X[:, j], X[:, k]).statistic
                if not np.isfinite(tau):
                    tau = 0.0
                # Convert scipy's tie-corrected tau-b to the paper's pair-average tau-a.
                tau *= np.sqrt(untied[j] * untied[k]) / pairs
                correlation[j, k] = correlation[k, j] = np.sin(np.pi * tau / 2)
        covariance = np.sqrt(variance[:, None] * variance[None, :]) * correlation
        covariance = _nearest_psd_max_norm(covariance)
    covariance = (covariance + covariance.T) / 2
    return mean, covariance + ridge * np.eye(d)


def clime_precision(covariance, regularization=0.1):
    """Columnwise CLIME linear programs, then smaller-magnitude symmetrization.

    Solve min ||w||1 subject to ||covariance*w-e_j||inf <= regularization.
    """
    from scipy.optimize import linprog

    d = covariance.shape[0]
    precision = np.empty((d, d))
    constraints = np.block([[covariance, -covariance], [-covariance, covariance]])
    for column in range(d):
        target = np.eye(d)[:, column]
        solution = linprog(
            np.ones(2 * d),
            A_ub=constraints,
            b_ub=np.r_[target + regularization, -target + regularization],
            bounds=(0, None),
            method="highs",
        )
        if not solution.success:
            raise RuntimeError(f"QUADRO CLIME precision fit failed: {solution.message}")
        precision[:, column] = solution.x[:d] - solution.x[d:]
    return np.where(np.abs(precision) <= np.abs(precision.T), precision, precision.T)


def estimate_kurtosis(X, mean, covariance, precision="clime", clime_lambda=0.1):
    """Section 4.3 radial moment formula, with CLIME or a covariance inverse."""
    centered = X - mean
    inverse = (
        clime_precision(covariance, clime_lambda)
        if precision == "clime"
        else np.linalg.solve(covariance, np.eye(covariance.shape[0]))
    )
    radius_squared = np.einsum("ij,ij->i", centered, centered @ inverse)
    d = X.shape[1]
    return max(float(np.mean(radius_squared**2) / (d * (d + 2)) - 1), 0.0)


def projected_moments(omega, delta, mean, covariance, gamma):
    """Proposition 2.1: mean and variance of Q(X)=X'omega X-2delta'X."""
    product = omega @ covariance
    trace = np.trace(product)
    shifted = omega @ mean - delta
    location = trace + mean @ omega @ mean - 2 * delta @ mean
    variance = (
        2 * (1 + gamma) * np.trace(product @ product)
        + gamma * trace**2
        + 4 * shifted @ covariance @ shifted
    )
    return float(location), max(float(variance), np.finfo(float).eps)
