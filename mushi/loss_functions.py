"""Loss functions"""

import jax.numpy as np


def prf(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""Poisson random field loss

    .. math::
        \sum_{i,j} \left( \mathbb{E}[X_{i, j}] - X_{i, j} \log \mathbb{E}[X_{i, j}] \right)

    Args:
        E: expectation :math:`\mathbb{E}[\mathbf X]`
        X: data :math:`\mathbf X`

    """
    return (E - X * np.log(E)).sum()


def dkl(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""Generalized Kullback-Liebler divergence, a Bregman divergence (ignores
    constant term)

    .. math::
        \sum_{i,j} \left(X_{i, j} \log \frac{X_{i, j}}{\mathbb{E}[X_{i, j}]} - X_{i, j} + \mathbb{E}[X_{i, j}] \right)

    Args:
        E: expectation :math:`\mathbb{E}[\mathbf X]`
        X: data :math:`\mathbf X`

    """
    return (X * np.log(X / E) - X + E).sum()


def lsq(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""Least-squares loss

    .. math::
        \frac{1}{2} \| \mathbb{E}[\mathbf X] - \mathbf X \|_F^2

    Args:
        E: expectation :math:`\mathbb{E}[\mathbf X]`
        X: data :math:`\mathbf X`

    """
    return (1 / 2) * ((E - X) ** 2).sum()
