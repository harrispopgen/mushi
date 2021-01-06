r"""Optimization functions.

"""

import numpy as np
from typing import Callable
import prox_tv as ptv
from typing import Tuple
from scipy.linalg import cholesky_banded, cho_solve_banded
from functools import lru_cache

def acc_prox_grad_method(x: np.ndarray,  # noqa: C901
                         g: Callable[[np.ndarray], np.float64],
                         grad_g: Callable[[np.ndarray], np.float64],
                         h: Callable[[np.ndarray], np.float64],
                         prox: Callable[[np.ndarray, np.float64], np.float64],
                         tol: np.float64 = 1e-10,
                         max_iter: int = 100,
                         s0: np.float64 = 1,
                         max_line_iter: int = 100,
                         gamma: np.float64 = 0.8,
                         verbose: bool = False) -> np.ndarray:
    r"""Nesterov accelerated proximal gradient method with backtracking line
    search [1]_.

    The optimization problem solved is:

    .. math::
        \arg\min_x g(x) + h(x)

    where :math:`g` is differentiable, and the proximal operator for :math:`h`
    is available.

    Args:
        x: initial point
        g: differentiable term in objective function
        grad_g: gradient of g
        h: non-differentiable term in objective function
        prox: proximal operator corresponding to h
        tol: relative tolerance in objective function for convergence
        max_iter: maximum number of proximal gradient steps
        s0: initial step size
        max_line_iter: maximum number of line search steps
        gamma: step size shrinkage rate for line search
        verbose: print convergence messages

    Returns:
        solution point

    References
    ----------
    .. [1] https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

    """
    # initialize step size
    s = s0
    # initialize momentum iterate
    q = x
    # initial objective value
    f = g(x) + h(x)
    if verbose:
        print(f'initial objective {f:.6e}', flush=True)
    for k in range(1, max_iter + 1):
        # evaluate differtiable part of objective at momentum point
        g1 = g(q)
        grad_g1 = grad_g(q)
        if not np.all(np.isfinite(grad_g1)):
            raise RuntimeError(f'invalid gradient at iteration {k + 1}: '
                               f'{grad_g1}')
        # store old iterate
        x_old = x
        # Armijo line search
        for line_iter in range(max_line_iter):
            # new point via prox-gradient of momentum point
            x = prox(q - s * grad_g1, s)
            # G_s(q) as in the notes linked above
            G = (1 / s) * (q - x)
            # test g(q - sG_s(q)) for sufficient decrease
            if g(q - s * G) <= (g1 - s * (grad_g1 * G).sum()
                                + (s / 2) * (G ** 2).sum()):
                # Armijo satisfied
                break
            else:
                # Armijo not satisfied
                s *= gamma  # shrink step size

        # update momentum point
        q = x + ((k - 1) / (k + 2)) * (x - x_old)

        if line_iter == max_line_iter - 1:
            print('warning: line search failed', flush=True)
            s = s0
        if not np.all(np.isfinite(x)):
            print('warning: x contains invalid values', flush=True)
        # terminate if objective function is constant within tolerance
        f_old = f
        f = g(x) + h(x)
        rel_change = np.abs((f - f_old) / f_old)
        if verbose:
            print(f'iteration {k}, objective {f:.3e}, '
                  f'relative change {rel_change:.3e}',
                  end='        \r', flush=True)
        if rel_change < tol:
            if verbose:
                print(f'\nrelative change in objective function {rel_change:.2g} '
                      f'is within tolerance {tol} after {k} iterations',
                      flush=True)
            break
        if k == max_iter:
            if verbose:
                print(f'\nmaximum iteration {max_iter} reached with relative '
                      f'change in objective function {rel_change:.2g}',
                      flush=True)

    return x


def three_op_prox_grad_method(x: np.ndarray,  # noqa: C901
                              g: Callable[[np.ndarray], np.float64],
                              grad_g: Callable[[np.ndarray], np.float64],
                              h1: Callable[[np.ndarray], np.float64],
                              prox1: Callable[[np.ndarray, np.float64],
                                              np.float64],
                              h2: Callable[[np.ndarray], np.float64],
                              prox2: Callable[[np.ndarray, np.float64],
                                              np.float64],
                              tol: np.float64 = 1e-10,
                              max_iter: int = 100,
                              s0: np.float64 = 1,
                              max_line_iter: int = 100,
                              gamma: np.float64 = 0.8,
                              ls_tol: np.float64 = 0,
                              verbose: bool = False) -> np.ndarray:
    r"""Three operator splitting proximal gradient method with backtracking line
    search [2]_.

    The optimization problem solved is:

    .. math::
        \arg\min_x g(x) + h_1(x) + h_2(x)

    where :math:`g` is differentiable, and the proximal operators for
    :math:`h_1` and :math:`h_2` are available.

    Args:
        x: initial point
        g: differentiable term in objective function
        grad_g: gradient of g
        h1: 1st non-differentiable term in objective function
        h2: 2nd non-differentiable term in objective function
        prox1: proximal operator corresponding to h1
        prox2: proximal operator corresponding to h2
        tol: relative tolerance in objective function for convergence
        max_iter: maximum number of proximal gradient steps
        s0: step size
        max_line_iter: maximum number of line search steps
        gamma: step size shrinkage rate for line search
        ls_tol: line search tolerance
        verbose: print convergence messages

    Returns:
        solution point

    References:
        .. [2] Pedregosa, Gidel, Adaptive Three Operator Splitting in
               Proceedings of the 35th International Conference on Machine
               Learning, Proceedings of Machine Learning Research., J. Dy, A.
               Krause, Eds. (PMLR, 2018), pp. 4085–4094.

    """

    # initial objective value
    s = s0
    z = x
    u = np.zeros_like(z)
    f = g(x) + h1(x) + h2(x)
    if verbose:
        print(f'initial objective {f:.6e}', flush=True)

    for k in range(1, max_iter + 1):
        # evaluate differentiable part of objective
        g1 = g(z)
        grad_g1 = grad_g(z)
        if not np.all(np.isfinite(grad_g1)):
            raise RuntimeError(f'invalid gradient at iteration {k + 1}: '
                               f'{grad_g1}')
        # store old iterate
        # x_old = x
        # Armijo line search
        for line_iter in range(max_line_iter):
            # new point via prox-gradient of momentum point
            x = prox1(z - s * (u + grad_g1), s)
            # quadratic approximation of objective
            Q = (g1 + (grad_g1 * (x - z)).sum()
                    + ((x - z) ** 2).sum() / (2 * s))
            if g(x) - Q <= ls_tol:
                # sufficient decrease satisfied
                break
            else:
                # sufficient decrease not satisfied
                s *= gamma  # shrink step size
        if line_iter == max_line_iter - 1:
            print('warning: line search failed', flush=True)

        # update z variables with 2nd prox
        z = prox2(x + s * u, s)
        # update u variables: dual variables
        u = u + (x - z) / s
        # grow step size
        s = min(s / gamma ** 2, s0)

        # TODO: convergence based on dual certificate
        if not np.all(np.isfinite(x)):
            print('warning: x contains invalid values', flush=True)
        # terminate if objective function is constant within tolerance
        f_old = f
        f = g(x) + h1(x) + h2(x)
        rel_change = np.abs((f - f_old) / f_old)
        if verbose:
            print(f'iteration {k}, objective {f:.3e}, '
                  f'relative change {rel_change:.3e}',
                  end='        \r', flush=True)
        if rel_change < tol:
            if verbose:
                print(f'\nrelative change in objective function {rel_change:.2g} '
                      f'is within tolerance {tol} after {k} iterations',
                      flush=True)
            break
        # if certificate < tol:
        #     print(f'certificate norm {certificate:.2g} '
        #           f'is within tolerance {tol} after {k} iterations')
        #     break
        if k == max_iter:
            if verbose:
                print(f'\nmaximum iteration {max_iter} reached with relative '
                      f'change in objective function {rel_change:.2g}', flush=True)

    return x


def trend_filter(y: np.ndarray, k: Tuple[np.float64], λ: Tuple[np.float64],
                       n_iter=100, ρ: Tuple[np.float64] = None) -> np.ndarray:
    r"""Mixed trend filtering via specialized ADMM as in [3]_, section 5.2.

    The optimization problem solved is:

    .. math::
        \arg\min_{\beta \in \mathbb{R}} \frac{1}{2} \|y - \beta\|_2^2 + \sum_{\ell=1}^r \lambda_\ell \| D^{k_\ell + 1} \beta \|_1

    where :math:`r` is the number of elements of k.

    Args:
        y: input signal vector, or matrix of column signals
        k: tuple of integer trend filter orders
        λ: tuple of penalties corresponding to each k
        n_iter: number of iterations of ADMM
        ρ: ADMM convergence parameter

    Returns:
        trend filtered solution

    References:
        .. [3] Aaditya Ramdas and Ryan J. Tibshirani.
               Fast and flexible admm algorithms for trend filtering.
               Journal of Computational and Graphical Statistics 25.3 (2016): 839-858.
    """
    if y.ndim == 1:
        y = y[:, np.newaxis]

    # default ADMM parameters used in referenced paper
    if ρ is None:
        ρ = λ

    r = len(k)
    n = len(y)

    D, DTD = D_DTD(n, k)
    c = choleskify(n, k, ρ)

    # initialize solution point
    β = np.zeros_like(y)
    # initialize Lagrangian dual variables
    α = [np.zeros_like(y)[:(n - k)] for k in k]
    u = [np.zeros_like(y)[:(n - k)] for k in k]

    # ADMM iterations
    for iter in range(n_iter):
        β = cho_solve_banded((c, False),
                             y + sum(ρ[l] * D[l].T @ (α[l] + u[l]) for l in range(r)),
                             check_finite=False)
        if iter + 1 == n_iter:
            break
        for l in range(r):
            Dβ = D[l] @ β
            for j in range(β.shape[1]):
                α[l][:, j] = ptv.tv1_1d(Dβ[:, j] - u[l][:, j], λ[l] / ρ[l])
            u[l] += α[l] - Dβ

    # squeeze out singleton dimension if input is a vector
    return np.squeeze(β)


# cached helper functions for trend filtering
@lru_cache()
def D_DTD(n, k):
    """difference operator for each order in k"""
    D = np.eye(n, k=0) - np.eye(n, k=-1)
    D = tuple(np.linalg.matrix_power(D, k)[k:] for k in k)

    return tuple(zip(*((D[l], D[l].T @ D[l]) for l in range(len(k)))))


@lru_cache(maxsize=None)
def choleskify(n, k, ρ):
    """cholesky decomposition needed for linear solves in trend estimate"""
    DTD = D_DTD(n, k)[1]
    A = np.eye(n) + sum(ρ[l] * DTD[l] for l in range(len(k)))
    # A is a banded Hermitian positive definite matrix with upper/lower bandwidth bw
    # express in upper diagonal ordered form
    bw = max(k)
    Ab = np.zeros((bw + 1, A.shape[1]))
    for u in range(bw + 1):
        Ab[-(1 + u), u:] = np.diag(A, k=u)

    return cholesky_banded(Ab, check_finite=False)
