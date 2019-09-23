#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import binom

import histories

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def C(n: int) -> np.ndarray:
    '''The C matrix defined in the text

    n: number of sampled haplotypes
    '''
    W1 = np.zeros((n - 1, n - 1))
    W2 = np.zeros((n - 1, n - 1))
    b = np.arange(1, n - 1 + 1)
    # j = 2
    W1[:, 0] = 6 / (n + 1)
    W2[:, 0] = 0
    # j = 3
    W1[:, 1] = 10 * (5 * n - 6 * b - 4) / (n + 2) / (n + 1)
    W2[:, 1] = (20 * (n - 2)) / (n+1) / (n+2)
    for j in range(2, n - 1):
        col = j - 2
        # procedurally generated by Zeilberger's algorithm in Mathematica
        W1[:, col + 2] = -((-((-1 + j)*(1 + j)**2*(3 + 2*j)*(j - n)*(4 + 2*j - 2*b*j + j**2 - b*j**2 + 4*n + 2*j*n + j**2*n)*W1[:, col]) - (-1 + 2*j)*(3 + 2*j)*(-4*j - 12*b*j - 4*b**2*j - 6*j**2 - 12*b*j**2 - 2*b**2*j**2 - 4*j**3 + 4*b**2*j**3 - 2*j**4 + 2*b**2*j**4 + 4*n + 2*j*n - 6*b*j*n + j**2*n - 9*b*j**2*n - 2*j**3*n - 6*b*j**3*n - j**4*n - 3*b*j**4*n + 4*n**2 + 6*j*n**2 + 7*j**2*n**2 + 2*j**3*n**2 + j**4*n**2)*W1[:, col + 1])/(j**2*(2 + j)*(-1 + 2*j)*(1 + j + n)*(3 + b + j**2 - b*j**2 + 3*n + j**2*n)))
        W2[:, col + 2] = ((-1 + j)*(1 + j)*(2 + j)*(3 + 2*j)*(j - n)*(1 + j - n)*(1 + j + n)*W2[:, col] + (-1 + 2*j)*(3 + 2*j)*(1 + j - n)*(j + n)*(2 - j - 2*b*j - j**2 - 2*b*j**2 + 2*n + j*n + j**2*n)*W2[:, col + 1])/((-1 + j)*j*(2 + j)*(-1 + 2*j)*(j - n)*(j + n)*(1 + j + n))

    return W1 - W2

def M(n: int, η: histories.η) -> np.ndarray:
    '''The M matrix defined in the text

    n: number of sampled haplotypes
    η: demographic history
    '''
    t, y = η.arrays()
    # epoch durations
    s = np.diff(t)
    u = np.exp(-s / y)
    u = np.concatenate((np.array([1]), u))

    binom_vec = binom(np.arange(2, n + 1), 2)

    return np.diag(1 / binom_vec) \
        @ np.cumprod(u[np.newaxis, :-1],
                     axis=1) ** binom_vec[:, np.newaxis] \
        @ (np.eye(len(y), k=0) - np.eye(len(y), k=-1)) \
        @ np.diag(y)


def ℓ(Z: np.ndarray, X: np.ndarray, L: np.ndarray,
      grad: bool = False) -> np.float:
    '''Poisson random field log-likelihood of history

    Z: mutation spectrum history matrix (μ.Z)
    X: k-SFS data
    L: model matrix
    grad: flag to also return gradient wrt Z
    '''
    Ξ = L @ Z
    ℓ = poisson.logpmf(X, Ξ).sum()
    if grad:
        dℓdZ = L.T @ (X / Ξ - 1)
        return np.array([ℓ, dℓdZ])
    else:
        return ℓ


def d_kl(Z: np.ndarray, X: np.ndarray, L: np.ndarray,
         grad: bool = False) -> float:
    '''Kullback-Liebler divergence between normalized SFS and its
    expectation under history
    ignores constant term

    Z: mutation spectrum history matrix (μ.Z)
    X: k-SFS data
    L: model matrix
    grad: flag to also return gradient wrt Z
    '''
    X_normalized = X / X.sum(axis=0)
    Ξ = L @ Z
    Ξ_normalized = Ξ / Ξ.sum(axis=1, keepdims=True)
    d_kl = (-X_normalized * np.log(Ξ_normalized)).sum()
    if grad:
        grad_d_kl = -L.T @ ((X_normalized / Ξ) * (1 - Ξ_normalized))
        return np.array([d_kl, grad_d_kl])
    else:
        return d_kl

def lsq(Z: np.ndarray, X: np.ndarray, L: np.ndarray,
        grad: bool = False) -> float:
    '''least-squares loss between SFS and its expectation under history

    Z: mutation spectrum history matrix (μ.Z)
    X: k-SFS data
    L: model matrix
    grad: flag to also return gradient wrt μ
    '''
    if self.X is None:
        raise ValueError('use simulate() to generate data first')
    Ξ = self.L @ Z
    lsq = (1 / 2) * ((Ξ - self.X) ** 2).sum()
    if grad:
        grad_lsq = self.L.T @ (Ξ - self.X)
        return np.array([lsq, grad_lsq])
    else:
        return lsq
