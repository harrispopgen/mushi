r"""Loss functions for measuring goodness of fit.

Each loss function takes an expected data matrix :math:`\mathbb{E}[\mathbf X]`
and an observed data matrix :math:`\mathbf X`, and returns a loss value.
Higher loss means worse fit between :math:`\mathbf X` and
:math:`\mathbb{E}[\mathbf X]`.

Examples
--------

>>> import mushi.loss_functions as lf
>>> import numpy as np

Define expected data matrix :math:`\mathbb{E}[\mathbf X]` and an observed data
matrix :math:`\mathbf X` as :math:`10\times 10` arrays of ones (for this trivial
example):

>>> E = np.ones((10, 10))
>>> X = np.ones((10, 10))

Compute various losses:

- Poisson random field

  >>> lf.prf(E, X)
  DeviceArray(100., dtype=float64)

- Generalized Kullback-Leibler divergence

  >>> lf.dkl(E, X)
  DeviceArray(0., dtype=float64)

- Least squares

  >>> lf.lsq(E, X)
  0.0
"""

import jax.numpy as np


def prf(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""Poisson random field loss.

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
    r"""Least-squares loss.

    .. math::
        \frac{1}{2} \| \mathbb{E}[\mathbf X] - \mathbf X \|_F^2

    Args:
        E: expectation :math:`\mathbb{E}[\mathbf X]`
        X: data :math:`\mathbf X`
    """
    return (1 / 2) * ((E - X) ** 2).sum()
