#! /usr/bin/env python
# -*- coding: utf-8 -*-

from jax.config import config
config.update('jax_enable_x64', True)
# config.update('jax_debug_nans', True)
import numpy as onp
import jax.numpy as np
from jax import jit, grad
from jax.ops import index, index_update
from scipy.special import binom
from scipy.stats import poisson, chi2
from scipy.optimize import minimize, OptimizeResult
import prox_tv as ptv

from typing import List

import histories
import utils

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


class kSFS():
    """The kSFS model described in the text"""

    def __init__(self, η: histories.η = None, μ: histories.μ = None,
                 X: np.ndarray = None, n: int = None,
                 mutation_types: List[str] = None):
        u"""Sample frequency spectrum

        η: demographic history
        X: observed k-SFS matrix (optional)
        n: number of haplotypes (optional)
        mutation_types: names of X columns
        """
        self.η = η
        self.μ = μ
        if X is not None:
            self.X = X
            self.n = len(X) + 1
            if mutation_types is not None:
                self.mutation_types = pd.Index(mutation_types,
                                               name='mutation type')
            else:
                self.mutation_types = pd.Index(range(1, self.X.shape[1] + 1),
                                               name='mutation type')
        elif not n:
            raise ValueError('either x or n must be specified')
        else:
            self.n = n
        self.C = utils.C(self.n)
        if self.η is None:
            self.M = None
            self.L = None
        else:
            t, y = self.η.arrays()
            self.M = utils.M(self.n, t, y)
            self.L = self.C @ self.M

    def tmrca_cdf(self) -> onp.ndarray:
        """The CDF of the TMRCA of at each change point"""
        if self.η is None:
            raise ValueError('η(t) must be defined first')
        t, y = self.η.arrays()
        # epoch durations
        s = onp.diff(t)
        u = onp.exp(-s / y)
        u = onp.concatenate((onp.array([1]), u))
        # the A_2j are the product of this matrix
        # NOTE: using letter  "l" as a variable name to match text
        l = onp.arange(2, self.n + 1)[:, onp.newaxis]
        with onp.errstate(divide='ignore'):
            A2_terms = l * (l-1) / (l * (l-1) - l.T * (l.T-1))
        onp.fill_diagonal(A2_terms, 1)
        A2 = onp.prod(A2_terms, axis=0)

        return 1 - (A2[onp.newaxis, :]
                    @ onp.cumprod(u[onp.newaxis, 1:-1],
                                 axis=1) ** binom(onp.arange(2, self.n + 1), 2
                                                  )[:, onp.newaxis]).T

    def simulate(self, seed: int = None) -> None:
        """simulate a SFS under the Poisson random field model (no linkage)
        assigns simulated SFS to self.X

        seed: random seed (optional)
        """
        if self.η is None:
            raise ValueError('η(t) must be defined first')
        if self.μ is None:
            raise ValueError('μ(t) must be defined first')
        if not self.η.check_grid(self.μ):
            raise ValueError('η and μ histories must use the same time grid')
        onp.random.seed(seed)
        self.X = poisson.rvs(self.L @ self.μ.Z)
        self.mutation_types = self.μ.mutation_types

    def infer_constant(self, change_points: np.array, μ_0: np.float64 = None,
                       mask: np.ndarray = None):
        u"""infer constant η and μ

        change_points: epoch change points (times)
        μ_0: total mutation rate, if self.μ is None
        mask: array of bools, with True indicating exclusion of that frequency
        """
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        if mask is not None:
            assert len(mask) == self.X.shape[0], 'mask must have n-1 elements'
        # badness of fit
        if mask is not None:
            C = self.C[~mask, :]
            X = self.X[~mask, :]
        else:
            C = self.C
            X = self.X
        X_total = X.sum(1, keepdims=True)
        μ_total = histories.μ(change_points,
                              μ_0 * np.ones((len(change_points) + 1, 1)))
        t, z = μ_total.arrays()
        # number of segregating sites
        S = X_total.sum()
        # Harmonic number
        H = (1 / np.arange(1, self.n - 1)).sum()
        # constant MLE
        y = (S / 2 / H / μ_0) * np.ones(len(z))

        self.η = histories.η(change_points, y)
        self.μ = histories.μ(self.η.change_points,
                             μ_0 * (X.sum(axis=0, keepdims=True) / X.sum()) * np.ones((self.η.m, X.shape[1])),
                             mutation_types=self.mutation_types.values)
        self.M = utils.M(self.n, t, y)
        self.L = self.C @ self.M

    def coord_desc(self,
                   α_tv: np.float64 = 0,
                   α_spline: np.float64 = 0,
                   α_ridge: np.float64 = 0,
                   β_tv: np.float64 = 0,
                   β_spline: np.float64 = 0,
                   β_rank: np.float64 = 0,
                   β_ridge: np.float64 = 0,
                   γ: np.float64 = 0.8,
                   max_iter: int = 1000,
                   tol: np.float64 = 1e-4,
                   fit='prf',
                   hard=False,
                   mask: np.ndarray = None) -> np.ndarray:
        u"""perform one iteration of block coordinate descent to fit η and μ

        η(t) regularization parameters:
        - α_tv: fused LASSO regularization strength
        - α_spline: regularization strength for L2 on diff
        - α_ridge: L2 penalty for strong convexity

        μ(t) regularization parameters:
        - β_tv: fused LASSO regularization strength
        - β_spline: regularization strength for L2 on diff
        - β_rank: spectral regularization strength
        - β_ridge: regularization strength for Frobenius norm on Z (removes
                   scale ridge for η, μ and t, and promotes strong convexity)

        γ: step size shrinkage rate for line search
        max_iter: maximum number of proximal gradient descent steps
        tol: relative tolerance in objective function
        fit: loss function, 'prf' for Poisson random field, 'kl' for
             Kullback-Leibler divergence, 'lsq' for least-squares
        hard: hard Vs soft singular value thresholding
        mask: array of bools, with True indicating exclusion of that frequency

        returns cost function at final iterate
        """
        assert self.X is not None, 'use simulate() to generate data first'
        assert self.η is not None, 'must initialize e.g. with infer_constant()'

        if mask is not None:
            X = self.X[~mask, :]
        else:
            X = self.X

        t = self.η.arrays()[0]

        # badness of fit
        @jit
        def loss_func(y, Z):
            L = self.C @ utils.M(self.n, t, y)
            if mask is not None:
                L = L[~mask, :]
            if fit == 'prf':
                return -utils.prf(Z, X, L)
            elif fit == 'kl':
                return utils.d_kl(Z, X, L)
            elif fit == 'lsq':
                return utils.lsq(Z, X, L)
            else:
                raise ValueError(f'unrecognized fit argument {fit}')

        if α_tv > 0:
            def prox_update_y(y, s):
                """L1 prox operator"""
                return ptv.tv1_1d(y, s * α_tv)
        else:
            def prox_update_y(y, s):
                return y

        if β_tv > 0 and β_rank > 0:
            raise NotImplementedError('fused LASSO with l1 spectral '
                                      'regularization not available')
        elif β_tv > 0:
            def prox_update_Z(Z, s):
                """L1 prox operator on row dimension (oddly 1-based indexed in
                proxtv)
                """
                return ptv.tvgen(Z, [s * β_tv], [1], [1])
        elif β_rank > 0:
            if hard:
                def prox_update_Z(Z, s):
                    """l0 norm on singular values (hard-thresholding)"""
                    U, σ, Vt = np.linalg.svd(Z, full_matrices=False)
                    σ = index_update(σ, index[σ <= s * β_rank], 0)
                    Σ = np.diag(σ)
                    return U @ Σ @ Vt
            else:
                def prox_update_Z(Z, s):
                    """l1 norm on singular values (soft-thresholding)"""
                    U, σ, Vt = np.linalg.svd(Z, full_matrices=False)
                    Σ = np.diag(np.maximum(0, σ - s * β_rank))
                    return U @ Σ @ Vt
        else:
            def prox_update_Z(Z, s):
                """project onto positive orthant"""
                return np.clip(Z, 0, np.inf)

        # Accelerated proximal gradient descent: our cost function decomposes
        # as f = g + h, where g is differentiable and h is not.
        # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
        # We'll do block coordinate descent partitioned by y and Z

        # some matrices we'll need for the first difference penalties
        D = (np.eye(self.η.m, k=0) - np.eye(self.η.m, k=-1))
        # W matrix deals with boundary condition
        W = np.eye(self.η.m)
        W = index_update(W, index[0, 0], 0)
        D1 = W @ D  # 1st difference matrix

        @jit
        def g(y, Z):
            """differentiable piece of cost"""
            return loss_func(y, Z) \
                + (α_spline / 2) * ((D1 @ y) ** 2).sum() \
                + (α_ridge / 2) * (y ** 2).sum() \
                + (β_spline / 2) * ((D1 @ Z) ** 2).sum() \
                + (β_ridge / 2) * (Z ** 2).sum()

        @jit
        def h(y, Z):
            """nondifferentiable piece of cost"""
            σ = np.linalg.svd(Z, compute_uv=False)
            if hard:
                rank_penalty = np.linalg.norm(σ, 0)
            else:
                rank_penalty = np.linalg.norm(σ, 1)
            return α_tv * np.abs(D1 @ y).sum() + β_tv * np.abs(D1 @ Z).sum() + β_rank * rank_penalty

        # initial iterate
        y = self.η.y
        Z = self.μ.Z
        # initial loss as first element of f_trajectory we'll append to
        f_trajectory = [g(y, Z) + h(y, Z)]
        # max step size
        s0 = 1
        # max number of Armijo step size reductions
        max_line_iter = 100

        print('η block')
        #logy = utils.acc_prox_grad_descent(
        y = utils.three_op_prox_grad_descent(
                              yy,
                              jit(lambda y: g(y, Z)),
                              jit(grad(lambda y: g(y, Z))),
                              jit(lambda y: h(y, Z)),
                              prox_update_y,
                              tol=tol,
                              max_iter=max_iter,
                              s0=s0,
                              max_line_iter=max_line_iter,
                              γ=γ)

        print('μ block')
        #Z = utils.acc_prox_grad_descent(
        Z = utils.three_op_prox_grad_descent(
                              Z,
                              jit(lambda Z: g(y, Z)),
                              jit(grad(lambda Z: g(y, Z))),
                              jit(lambda Z: h(y, Z)),
                              prox_update_Z,
                              tol=tol,
                              max_iter=max_iter,
                              s0=s0,
                              max_line_iter=max_line_iter,
                              γ=γ,
                              nonneg=True)

        self.η = histories.η(self.η.change_points, y)
        self.M = utils.M(self.n, t, y)
        self.L = self.C @ self.M
        self.μ = histories.μ(self.η.change_points, Z,
                             mutation_types=self.mutation_types.values)
        return g(y, Z) + h(y, Z)

    def plot_total(self):
        """plot the total SFS"""
        if self.η is not None:
            if self.μ is not None:
                z = self.μ.Z.sum(1)
            else:
                z = np.ones_like(self.η.y)
            ξ = self.L.dot(z)
            plt.plot(range(1, self.n), ξ, c='C1', ls='--', label=r'$\xi$')
            ξ_lower = poisson.ppf(.025, ξ)
            ξ_upper = poisson.ppf(.975, ξ)
            plt.fill_between(range(1, self.n),
                             ξ_lower, ξ_upper,
                             facecolor='C1', alpha=0.25,
                             label='inner 95%\nquantile')
        x = self.X.sum(1, keepdims=True)
        plt.plot(range(1, len(x) + 1), x,
                 c='C0', ls='', marker='.', alpha=.25, label=r'data')
        plt.xlabel('sample frequency')
        plt.ylabel(r'number of variants')
        plt.xscale('log')
        plt.yscale('symlog')
        plt.tight_layout()

    def plot(self, type=None, normed: bool = False, **kwargs) -> None:
        """normed: flag to normalize to relative mutation intensity"""
        if self.μ is not None:
            Ξ = self.L @ self.μ.Z
        if normed:
            X = self.X / self.X.sum(1, keepdims=True)
            if self.μ is not None:
                Ξ = Ξ / Ξ.sum(1, keepdims=True)
            plt.ylabel('mutation type fraction')
        else:
            X = self.X
            plt.ylabel('number of variants')
        if type is not None:
            i = self.mutation_types.get_loc(type)
            X = X[:, i]
            if self.μ is not None:
                Ξ = Ξ[:, i]

        if self.μ is not None:
            plt.plot(range(1, self.n), X, ls='', marker='.', **kwargs)
            line_kwargs = kwargs
            if 'label' in line_kwargs:
                del line_kwargs['label']
            plt.gca().set_prop_cycle(None)
            plt.plot(range(1, self.n), Ξ, **line_kwargs)
        else:
            plt.plot(range(1, self.n), X, **kwargs)
        plt.xlabel('sample frequency')
        plt.xscale('symlog')
        if 'label' in kwargs:
            plt.legend()
        plt.tight_layout()

    def clustermap(self, linthresh=1, **kwargs):
        u"""clustermap with mixed linear-log scale color bar

        μ: inferred mutation spectrum history, χ^2 values are shown if not None
        linthresh: the range within which the plot is linear (when μ = None)
        kwargs: additional keyword arguments passed to pd.clustermap
        """
        if self.μ is None:
            Z = self.X / self.X.sum(axis=1, keepdims=True)
            Z = Z / Z.mean(0, keepdims=True)
            cbar_label = 'mutation type\nenrichment'
        else:
            Ξ = self.L @ self.μ.Z
            Z = (self.X - Ξ) ** 2 / Ξ
            cbar_label = '$\\chi^2$'
            χ2_total = Z.sum()
            p = chi2(np.prod(Z.shape)).sf(χ2_total)
            print(f'χ\N{SUPERSCRIPT TWO} goodness of fit {χ2_total}, '
                  f'p = {p}')
        df = pd.DataFrame(data=Z, index=pd.Index(range(1, self.n),
                                                 name='sample frequency'),
                          columns=self.mutation_types)
        g = sns.clustermap(df, row_cluster=False, metric='correlation',
                           cbar_kws={'label': cbar_label}, **kwargs)
        g.ax_heatmap.set_yscale('symlog')
        return g
