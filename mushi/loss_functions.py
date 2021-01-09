"""Loss functions"""

import numpy as onp
import jax.numpy as np
from jax.ops import index, index_update
from typing import List


def prf(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""Poisson random field loss

    Args:
        E: expectation :math:`E[X]`
        X: data

    """
    return (E - X * np.log(E)).sum()


def dkl(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""generalized Kullback-Liebler divergence, a Bregman divergence (ignores
    constant term)

    Args:
        E: expectation :math:`E[X]`
        X: data

    """
    return (X * np.log(X / E) - X + E).sum()


def lsq(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""Least-squares loss

    Args:
        E: expectation :math:`E[X]`
        X: data

    """
    return (1 / 2) * ((E - X) ** 2).sum()
