r"""Optimization functions.

"""

import jax.numpy as np
from typing import Callable


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
        \min_x g(x) + h(x)

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
        \min_x g(x) + h_1(x) + h_2(x)

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
