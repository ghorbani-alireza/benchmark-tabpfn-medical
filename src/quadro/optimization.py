"""Matrix-free ADMM for the paper's constrained L1 QUADRO objective (equation 8)."""

import warnings
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.exceptions import ConvergenceWarning


def smooth_gradient(means, covariances, gamma, kappa, omega, delta):
    """Gradient of the shared unpenalized projected-variance objective."""
    quadratic, linear = np.zeros_like(omega), np.zeros_like(delta)
    for weight, mean, covariance in zip((1.0, kappa), means, covariances):
        shifted = omega @ mean - delta
        cross = np.outer(covariance @ shifted, mean)
        quadratic += weight * (
            4 * (1 + gamma) * covariance @ omega @ covariance
            + 2 * gamma * np.trace(omega @ covariance) * covariance
            + 4 * (cross + cross.T)
        )
        linear -= weight * 8 * covariance @ shifted
    return quadratic, linear


def solve_quadro(
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
):
    """Minimize L0+kappa*L1+lambda_Q*|omega|1+lambda_L*|delta|1.

    Constrain E[Q|0]-E[Q|1]=1 and omega=omega.T. This sign convention
    gives class 1 a lower projection; the Rayleigh objective is sign invariant.
    Full symmetric matrix coordinates count both off-diagonal entries in |omega|1.
    No d^4 Hessian or polynomial design matrix is constructed.
    """
    d = len(means[0])
    size = d * d + d

    def unpack(vector):
        omega = vector[: d * d].reshape(d, d)
        return (omega + omega.T) / 2, vector[d * d :]

    def pack(omega, delta):
        return np.concatenate([omega.ravel(), delta])

    weights = (1.0, kappa)
    contrast_matrix = (
        covariances[0]
        + np.outer(means[0], means[0])
        - covariances[1]
        - np.outer(means[1], means[1])
    )
    contrast = pack(contrast_matrix, -2 * (means[0] - means[1]))
    if np.linalg.norm(contrast) < 1e-12:
        raise ValueError("QUADRO cannot separate identical estimated class moments")

    def gradient(vector):
        omega, delta = unpack(vector)
        quadratic, linear = smooth_gradient(
            means, covariances, gamma, kappa, omega, delta
        )
        return pack(quadratic, linear)

    operator = LinearOperator((size, size), matvec=lambda v: gradient(v) + rho * v)

    # Jacobi preconditioning stabilizes small-sample/high-dimensional covariance fits.
    diagonal_quadratic = np.zeros((d, d))
    diagonal_linear = np.zeros(d)
    for weight, mean, covariance in zip(weights, means, covariances):
        diagonal = np.diag(covariance)
        diagonal_quadratic += weight * (
            2 * (1 + gamma) * (np.outer(diagonal, diagonal) + covariance**2)
            + 2 * gamma * covariance**2
            + 2 * (np.outer(diagonal, mean**2) + np.outer(mean**2, diagonal))
            + 4 * covariance * np.outer(mean, mean)
        )
        diagonal_linear += weight * 8 * diagonal
    diagonal = pack(diagonal_quadratic, diagonal_linear) + rho
    preconditioner = LinearOperator((size, size), matvec=lambda v: v / diagonal)

    def linear_solve(rhs, initial=None):
        solution, info = cg(
            operator,
            rhs,
            x0=initial,
            M=preconditioner,
            rtol=min(1e-7, tol / 10),
            atol=0,
            maxiter=max(1000, size * 10),
        )
        if info != 0:
            raise RuntimeError(
                f"QUADRO inner conjugate-gradient solver failed (info={info})"
            )
        return solution

    constraint_direction = linear_solve(contrast)
    denominator = contrast @ constraint_direction
    penalties = pack(np.full((d, d), lambda_quadratic), np.full(d, lambda_linear))
    sparse = np.zeros(size)
    dual = np.zeros(size)
    primal = np.zeros(size)
    converged = False
    for iteration in range(1, max_iter + 1):
        previous = sparse.copy()
        unconstrained = linear_solve(rho * (sparse - dual), primal)
        primal = unconstrained + constraint_direction * (
            (1 - contrast @ unconstrained) / denominator
        )
        shifted = primal + dual
        sparse = np.sign(shifted) * np.maximum(np.abs(shifted) - penalties / rho, 0)
        dual += primal - sparse
        primal_residual = np.linalg.norm(primal - sparse)
        dual_residual = rho * np.linalg.norm(sparse - previous)
        limit = tol * max(1.0, np.linalg.norm(primal), np.linalg.norm(sparse))
        if max(primal_residual, dual_residual) <= limit:
            converged = True
            break
    else:
        warnings.warn(
            "QUADRO optimizer did not converge; increase max_iter or adjust rho",
            ConvergenceWarning,
        )
    # Preserve sparsity and normalize the equality after numerical convergence.
    normalization = contrast @ sparse
    if abs(normalization) < 1e-12:
        raise RuntimeError("QUADRO optimizer produced an infeasible zero direction")
    omega, delta = unpack(sparse / normalization)
    residual = max(primal_residual, dual_residual)
    return omega, delta, iteration, residual, converged
