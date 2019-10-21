#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import numpy as np
from scipy.special import binom
from scipy.stats import poisson, chi2
from scipy.optimize import minimize, Bounds, OptimizeResult
import prox_tv as ptv

import histories
import utils

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


class kSFS():
    '''The kSFS model described in the text'''

    def __init__(self, η: histories.η = None, μ: histories.μ = None,
                 X: np.ndarray = None, n: int = None,
                 mutation_types: List[str] = None):
        '''Sample frequency spectrum

        η: demographic history
        X: observed k-SFS matrix (optional)
        n: number of haplotypes (optional)
        mutation_types: names of X columns
        '''
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
            self.L = (self.C @ self.M).astype(np.float64)

    def tmrca_cdf(self) -> np.ndarray:
        '''The cdf of the TMRCA of at each change point'''
        if self.η is None:
            raise ValueError('η(t) must be defined first')
        t, y = self.η.arrays()
        # epoch durations
        s = np.diff(t)
        u = np.exp(-s / y)
        u = np.concatenate((np.array([1]), u))
        # the A_2j are the product of this matrix
        # NOTE: using letter  "l" as a variable name to match text
        l = np.arange(2, self.n + 1)[:, np.newaxis]
        with np.errstate(divide='ignore'):
            A2_terms = l * (l-1) / (l * (l-1) - l.T * (l.T-1))
        np.fill_diagonal(A2_terms, 1)
        A2 = np.prod(A2_terms, axis=0)

        return 1 - (A2[np.newaxis, :]
                    @ np.cumprod(u[np.newaxis, 1:-1],
                                 axis=1) ** binom(np.arange(2, self.n + 1), 2
                                                  )[:, np.newaxis]).T

    def simulate(self, seed: int = None) -> None:
        '''simulate a SFS under the Poisson random field model (no linkage)
        assigns simulated SFS to self.X

        seed: random seed (optional)
        '''
        if self.η is None:
            raise ValueError('η(t) must be defined first')
        if self.μ is None:
            raise ValueError('μ(t) must be defined first')
        if not self.η.check_grid(self.μ):
            raise ValueError('η and μ histories must use the same time grid')
        np.random.seed(seed)
        self.X = poisson.rvs(self.L @ self.μ.Z)
        self.mutation_types = self.μ.mutation_types

    def infer_η(self, change_points: np.array = None, fit='prf',
                λ_tv: np.float64 = 0,
                μ_0: np.float64 = 1,
                mask: np.ndarray = None, **kwargs) -> OptimizeResult:
        '''infer η. Either self.μ is not None or change_points is not None. If
        the latter, fit using the total (row-summed) SFS and a unit constant
        total mutation rate history

        change_points: epoch change points (times)
        λ_tv: fused LASSO regularization strength
        μ_0: total mutation rate
        fit: loss function, 'prf' for Poisson random field, 'kl' for
             Kullback-Leibler divergence, 'lsq' for least-squares
        mask: array of bools, with True indicating exclusion of that frequency
        kwargs: key word arguments passed to scipy.optimize.minimize
        '''
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        assert λ_tv >= 0, 'λ_tv must be nonnegative'
        assert (self.μ is None) != (change_points is None)
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
        if self.μ is None:
            μ_total = histories.μ(change_points,
                                  μ_0 * np.ones((len(change_points) + 1, 1)))
        else:
            change_points = self.μ.change_points
            μ_total = histories.μ(change_points,
                                  self.μ.Z.sum(1, keepdims=True))
        t, z = μ_total.arrays()
        if fit == 'prf':
            def loss_func(y, **kwargs):
                L = (C @ utils.M(self.n, t, y)).astype(np.float64)
                return -utils.prf(μ_total.Z, X_total, L, **kwargs)
        elif fit == 'kl':
            def loss_func(y, **kwargs):
                L = (C @ utils.M(self.n, t, y)).astype(np.float64)
                return utils.d_kl(μ_total.Z, X_total, L, **kwargs)
        elif fit == 'lsq':
            def loss_func(y, **kwargs):
                L = (C @ utils.M(self.n, t, y)).astype(np.float64)
                return utils.lsq(μ_total.Z, X_total, L, **kwargs)
        else:
            raise ValueError(f'unrecognized fit argument {fit}')

        if self.η is None:
            # number of segregating sites
            S = X_total.sum()
            # Harmonic number
            H = (1 / np.arange(1, self.n - 1)).sum()
            # constant MLE
            y = (S / 2 / H / μ_0) * np.ones(len(z))
        else:
            y = self.η.y
        Λ = 1 / y

        def f(logΛ):
            y = 1 / np.exp(logΛ)
            return loss_func(y) + (λ_tv / 2) * (np.diff(y) ** 2).sum()

        result = minimize(f, np.log(Λ),
                          **kwargs)
        y = 1 / np.exp(result.x)
        self.η = histories.η(change_points, y)
        self.μ = histories.μ(self.η.change_points,
                             μ_0 * (X.sum(axis=0, keepdims=True) / X.sum()) * np.ones((self.η.m, X.shape[1])),
                             mutation_types=self.mutation_types.values)
        self.M = utils.M(self.n, t, y)
        self.L = (self.C @ self.M).astype(np.float64)
        return result

    def constant_μ_MLE(self, mask: np.ndarray = None):
        '''gives the MLE for a constant μ history

        mask: array of bools, with True indicating exclusion of that frequency
        '''
        if self.η is None:
            raise ValueError('η(t) must be defined first')
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        if mask is not None:
            L = self.L[~mask, :]
            X = self.X[~mask, :]
        else:
            L = self.L
            X = self.X
        z0 = X.sum(axis=0) / np.sum(L)
        self.μ = histories.μ(self.η.change_points,
                             z0[np.newaxis, :] * np.ones((self.η.m, 1)),
                             mutation_types=self.mutation_types.values)

    def infer_μ(self,
                λ_tv: np.float64 = 0, α_tv: np.float64 = 0,
                λ_r: np.float64 = 0,  α_r: np.float64 = 0,
                γ: np.float64 = 0.8, max_iter: int = 1000,
                tol: np.float64 = 1e-4, fit='prf',
                hard=False,
                mask: np.ndarray = None) -> np.ndarray:
        '''infer μ

        λ_tv: fused LASSO regularization strength
        α_tv: relative penalty on L1 vs L2 in fused LASSO
        λ_r: spectral (rank) regularization strength
        α_r: relative penalty on L1 vs L2 in spectral regularization
        γ: step size shrinkage rate for line search
        max_iter: maximum number of proximal gradient descent iterations
        tol: relative tolerance in objective function
        fit: loss function, 'prf' for Poisson random field, 'kl' for
             Kullback-Leibler divergence, 'lsq' for least-squares
        hard: hard Vs soft singular value thresholding
        mask: array of bools, with True indicating exclusion of that frequency

        returns array of cost function at each iterate
        '''
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        assert λ_tv >= 0, 'λ_tv must be nonnegative'
        assert λ_r >= 0, 'λ_r must be nonnegative'
        assert 0 <= α_tv <= 1, 'α_tv must be in the unit interval'
        assert 0 <= α_r <= 1, 'α_r must be in the unit interval'
        if mask is not None:
            assert len(mask) == self.X.shape[0], 'mask must have n-1 elements'
        # badness of fit
        if mask is not None:
            L = self.L[~mask, :]
            X = self.X[~mask, :]
        else:
            L = self.L
            X = self.X
        if fit == 'prf':
            def loss_func(Z, **kwargs):
                return -utils.prf(Z, X, L, **kwargs)
        elif fit == 'kl':
            def loss_func(Z, **kwargs):
                return utils.d_kl(Z, X, L, **kwargs)
        elif fit == 'lsq':
            def loss_func(Z, **kwargs):
                return utils.lsq(Z, X, L, **kwargs)
        else:
            raise ValueError(f'unrecognized fit argument {fit}')
        if λ_tv * α_tv > 0 and λ_r * α_r > 0:
            raise NotImplementedError('fused LASSO with l1 spectral '
                                      'regularization not available')
        elif λ_tv * α_tv > 0:
            def prox_update(Z, s):
                '''L1 prox operator on row dimension (oddly 1-based indexed in
                proxtv) with weight λα
                '''
                return ptv.tvgen(Z, [s * λ_tv * α_tv], [1], [1])
        elif λ_r * α_r > 0:
            if hard:
                def prox_update(Z, s):
                    '''l0 norm on singular values (hard-thresholding)
                    '''
                    U, σ, Vt = np.linalg.svd(Z, full_matrices=False)
                    σ[σ <= s * λ_r * α_r] = 0.0
                    Σ = np.diag(σ)
                    return U @ Σ @ Vt
            else:
                def prox_update(Z, s):
                    '''l1 norm on singular values (soft-thresholding)
                    '''
                    U, σ, Vt = np.linalg.svd(Z, full_matrices=False)
                    Σ = np.diag(np.maximum(0, σ - s * λ_r * α_r))
                    return U @ Σ @ Vt
        else:
            def prox_update(Z, s):
                return Z

        # Accelerated proximal gradient descent: our cost function decomposes
        # as f = g + h, where g is differentiable and h is not.
        # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
        # some matrices we'll need for the first difference penalties
        D = (np.eye(self.η.m, k=0) - np.eye(self.η.m, k=-1))
        W = np.eye(self.η.m)
        W[0, 0] = 0  # W matrix deals with boundary condition
        D1 = W @ D  # 1st difference matrix
        D2 = D1.T @ D1  # square of 1st difference matrix (Laplacian)

        def g(Z, grad=False):
            '''differentiable piece of cost'''
            if grad:
                loss, grad_loss = loss_func(Z, grad=True)
            else:
                loss = loss_func(Z)
            g = loss \
                + (λ_tv / 2) * (1 - α_tv) * ((D1 @ Z) ** 2).sum() \
                + (λ_r / 2) * (1 - α_r) * (Z ** 2).sum()
            if grad:
                grad_g = grad_loss + λ_tv * (1 - α_tv) * D2 @ Z \
                                     + λ_r * (1 - α_r) * Z
                return g, grad_g
            return g

        def h(Z):
            '''nondifferentiable piece of cost'''
            σ = np.linalg.svd(Z, compute_uv=False)
            if hard:
                rank_penalty = np.linalg.norm(σ, 0)
            else:
                rank_penalty = np.linalg.norm(σ, 1)
            return λ_tv * α_tv * np.abs(D1 @ Z).sum() \
                + λ_r * α_r * rank_penalty

        def f(Z):
            '''cost'''
            return g(Z) + h(Z)

        # # initialize using constant μ history MLE
        # μ = self.constant_μ_MLE(mask)
        Z = self.μ.Z  # current iterate
        Q = self.μ.Z  # momentum iterate
        # initial loss
        f_trajectory = [f(Z)]
        # initialize step size
        s0 = 1  # max step size
        s = s0  # current step size
        # max number of Armijo step size reductions
        max_line_iter = 100
        for k in range(1, max_iter + 1):
            # evaluate smooth part of loss at momentum point
            g1, grad_g1 = g(Q, grad=True)
            # store old iterate
            Z_old = Z
            # Armijo line search
            for line_iter in range(max_line_iter):
                if not np.all(np.isfinite(grad_g1)):
                    raise RuntimeError(f'invalid gradient at step {k}, line '
                                       f'search step {line_iter}: {grad_g1}')
                # new point via prox-gradient of momentum point
                Z = prox_update(Q - s * grad_g1, s)
                if not np.all(Z > 0):
                    print(f'warning: Z contains negative values after prox')
                    Z = np.clip(Z, 1e-6, np.inf)
                # G_s(Q) as in the notes linked above
                G = (1 / s) * (Q - Z)
                # test g(Q - sG_s(Q)) for sufficient decrease
                if g(Q - s * G) <= (g1 - s * (grad_g1 * G).sum()
                                       + (s / 2) * (G ** 2).sum()):
                    # Armijo satisfied
                    break
                else:
                    # Armijo not satisfied
                    s *= γ  # shrink step size
            # update momentum term
            Q = Z + ((k - 1) / (k + 2)) * (Z - Z_old)
            if line_iter == max_line_iter - 1:
                print('warning: line search failed')
                s = s0
            if not np.all(np.isfinite(Z)):
                print(f'warning: Z contains invalid values')
            if not np.all(Z > 0):
                print(f'warning: Z contains negative values')
            # terminate if loss function is constant within tolerance
            f_trajectory.append(f(Z))
            rel_change = np.abs((f_trajectory[-1] - f_trajectory[-2])
                                / f_trajectory[-2])
            if rel_change < tol:
                print(f'relative change in loss function {rel_change:.2g} '
                      f'is within tolerance {tol} after {k} iterations')
                break
            if k == max_iter:
                print(f'maximum iteration {max_iter} reached with relative '
                      f'change in loss function {rel_change:.2g}')
        self.μ = histories.μ(self.η.change_points, Z,
                             mutation_types=self.mutation_types.values)
        return np.array(f_trajectory)

    def plot_total(self):
        '''plot the total SFS
        '''
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
        '''
        normed: flag to normalize to relative mutation intensity
        '''
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
            plt.plot(range(1, self.n), Ξ, **line_kwargs)
        else:
            plt.plot(range(1, self.n), X, **kwargs)
        plt.xlabel('sample frequency')
        plt.xscale('symlog')
        if 'label' in kwargs:
            plt.legend()
        plt.tight_layout()

    def clustermap(self, linthresh=1, **kwargs):
        '''clustermap with mixed linear-log scale color bar

        μ: inferred mutation spectrum history, χ^2 values are shown if not None
        linthresh: the range within which the plot is linear (when μ = None)
        kwargs: additional keyword arguments passed to pd.clustermap
        '''
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
