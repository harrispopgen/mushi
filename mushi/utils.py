r"""Utility functions

"""

import numpy as onp
import jax.numpy as np
from jax.ops import index, index_update


def C(n: int) -> onp.ndarray:
    r"""The combinatorial :math:`C` matrix defined in the paper's appendix

    Args:
        n: the number of sampled haplotypes :math:`n`

    Returns:
        :math:`(n-1)\times(n-1)` matrix

    """
    W1 = onp.zeros((n - 1, n - 1))
    W2 = onp.zeros((n - 1, n - 1))
    b = onp.arange(1, n - 1 + 1)
    # j = 2
    W1[:, 0] = 6 / (n + 1)
    W2[:, 0] = 0
    # j = 3
    W1[:, 1] = 10 * (5 * n - 6 * b - 4) / (n + 2) / (n + 1)
    W2[:, 1] = (20 * (n - 2)) / (n+1) / (n+2)
    for col in range(n - 3):
        # this cast is crucial for floating point precision
        j = onp.float64(col + 2)
        # procedurally generated by Zeilberger's algorithm in Mathematica
        W1[:, col + 2] = -((-((-1 + j)*(1 + j)**2*(3 + 2*j)*(j - n)*(4 + 2*j - 2*b*j + j**2 - b*j**2 + 4*n + 2*j*n + j**2*n)*W1[:, col]) - (-1 + 2*j)*(3 + 2*j)*(-4*j - 12*b*j - 4*b**2*j - 6*j**2 - 12*b*j**2 - 2*b**2*j**2 - 4*j**3 + 4*b**2*j**3 - 2*j**4 + 2*b**2*j**4 + 4*n + 2*j*n - 6*b*j*n + j**2*n - 9*b*j**2*n - 2*j**3*n - 6*b*j**3*n - j**4*n - 3*b*j**4*n + 4*n**2 + 6*j*n**2 + 7*j**2*n**2 + 2*j**3*n**2 + j**4*n**2)*W1[:, col + 1])/(j**2*(2 + j)*(-1 + 2*j)*(1 + j + n)*(3 + b + j**2 - b*j**2 + 3*n + j**2*n)))
        W2[:, col + 2] = ((-1 + j)*(1 + j)*(2 + j)*(3 + 2*j)*(j - n)*(1 + j - n)*(1 + j + n)*W2[:, col] + (-1 + 2*j)*(3 + 2*j)*(1 + j - n)*(j + n)*(2 - j - 2*b*j - j**2 - 2*b*j**2 + 2*n + j*n + j**2*n)*W2[:, col + 1])/((-1 + j)*j*(2 + j)*(-1 + 2*j)*(j - n)*(j + n)*(1 + j + n))

    return W1 - W2


def M(n: int, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""The M matrix defined in the paper's appendix

    Args:
        n: the number of sampled haplotypes :math:`n`
        t: time grid, starting at zero and ending at np.inf
        y: population size in each epoch

    Returns:
        :math:`(n-1)\times m` matrix, where :math:`m` is the number of epochs
        (the length of the ``y`` argument)

    """
    # epoch durations
    s = np.diff(t)
    # we handle the final infinite epoch carefully to facilitate autograd
    u = np.exp(-s[:-1] / y[:-1])
    u = np.concatenate((np.array([1]), u, np.array([0])))

    n_range = np.arange(2, n + 1)
    binom_vec = n_range * (n_range - 1) / 2

    return np.exp(binom_vec[:, np.newaxis]
                  * np.cumsum(np.log(u[np.newaxis, :-1]), axis=1)
                  - np.log(binom_vec[:, np.newaxis])) \
        @ (np.eye(len(y), k=0) - np.eye(len(y), k=-1)) \
        @ np.diag(y)


def prf(Z: np.ndarray, X: np.ndarray, L: np.ndarray) -> np.float64:
    r"""Poisson random field log-likelihood of history (neglecting constant
    terms)

    Args:
        Z: mutation spectrum history matrix (μ.Z)
        X: :math:`k`-SFS data
        L: model matrix

    """
    Ξ = L @ Z
    return (X * np.log(Ξ) - Ξ).sum()


def d_kl(Z: np.ndarray, X: np.ndarray, L: np.ndarray) -> np.float64:
    r"""generalized Kullback-Liebler divergence (a Bregman divergence) between
    normalized SFS and its expectation under history (ignores constant term)

    Args:
        Z: mutation spectrum history matrix (μ.Z)
        X: k-SFS data
        L: model matrix

    """
    Ξ = L @ Z
    d_kl = (X * np.log(X / Ξ) - X + Ξ).sum()
    return d_kl


def lsq(Z: np.ndarray, X: np.ndarray, L: np.ndarray) -> float:
    r"""Least-squares loss between SFS and its expectation under history

    Args:
        Z: mutation spectrum history matrix (μ.Z)
        X: k-SFS data
        L: model matrix

    """
    Ξ = L @ Z
    lsq = (1 / 2) * ((Ξ - X) ** 2).sum()
    return lsq


def tmrca_sf(t: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """The survival function of the TMRCA at each time point

    Args:
        t: time grid (including zero and infinity)
        y: effective population size in each epoch
        n: number of sampled haplotypes

    """
    # epoch durations
    s = np.diff(t)
    logu = -s / y
    logu = np.concatenate((np.array([0]), logu))
    # the A_2j are the product of this matrix
    # NOTE: using letter  "l" as a variable name to match text
    l = onp.arange(2, n + 1)[:, onp.newaxis]
    with onp.errstate(divide='ignore'):
        A2_terms = l * (l - 1) / (l * (l - 1) - l.T * (l.T - 1))
    onp.fill_diagonal(A2_terms, 1)
    A2 = np.prod(A2_terms, axis=0)

    binom_vec = l * (l - 1) / 2

    result = np.zeros(len(t))
    result = index_update(result, index[:-1],
                          np.squeeze(A2[np.newaxis, :]
                                     @ np.exp(np.cumsum(logu[np.newaxis, :-1],
                                                        axis=1)) ** binom_vec))

    assert np.all(np.isfinite(result))

    return result
