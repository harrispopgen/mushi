#! /usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import numpy as np
from scipy.special import binom
from scipy.stats import poisson, chi2
import prox_tv as ptv

import histories
import utils

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


class kSFS():
    '''The kSFS model described in the text'''

    def __init__(self, η: histories.η, X: np.ndarray = None, n: int = None,
                 mutation_types: List[str] = None):
        '''Sample frequency spectrum

        η: demographic history
        X: observed k-SFS matrix (optional)
        n: number of haplotypes (optional)
        mutation_types: names of X columns
        '''
        self.η = η
        if X is not None:
            self.X = X
            self.n = len(X) + 1
            self.mutation_types = pd.Index(mutation_types,
                                           name='mutation type')
        elif not n:
            raise ValueError('either x or n must be specified')
        else:
            self.n = n
        self.L = utils.C(self.n) @ utils.M(self.n, self.η)

    def tmrca_cdf(self) -> np.ndarray:
        '''The cdf of the TMRCA of at each change point'''
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

    def simulate(self, μ: histories.μ, seed: int = None) -> None:
        '''simulate a SFS under the Poisson random field model (no linkage)
        assigns simulated SFS to self.X

        μ: mutation spectrum history
        seed: random seed (optional)
        '''
        if not self.η.check_grid(μ):
            raise ValueError('η and μ histories must use the same time grid')
        np.random.seed(seed)
        self.X = poisson.rvs(self.L @ μ.Z)
        self.mutation_types = μ.mutation_types

    def constant_μ_MLE(self) -> histories.μ:
        '''gives the MLE for a constant μ history'''
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        z0 = self.X.sum(axis=0) / np.sum(self.L)
        return histories.μ(self.η.change_points,
                           z0[np.newaxis, :] * np.ones((self.η.m, 1)),
                           mutation_types=self.mutation_types.values)

    def infer_μ(self,
                λ_tv: np.float64 = 0, α_tv: np.float64 = 0,
                λ_r: np.float64 = 0,  α_r: np.float64 = 0,
                γ: np.float64 = 0.8, max_iter: int = 1000,
                tol: np.float64 = 1e-4, fit='prf',
                bins: np.ndarray = None,
                hard=False,
                exclude_singletons: bool = False) -> histories.μ:
        '''return inferred μ history given the sfs and η history

        λ_tv: fused LASSO regularization strength
        α_tv: relative penalty on L1 vs L2 in fused LASSO
        λ_r: spectral (rank) regularization strength
        α_r: relative penalty on L1 vs L2 in spectral regularization
        γ: step size shrinkage rate for line search
        max_iter: maximum number of proximal gradient descent iterations
        tol: relative tolerance in objective function
        fit: fit function, 'prf' for Poisson random field, 'kl' for
             Kullback-Leibler divergence, 'lsq' for least-squares
        bins: SFS frequency bins
        hard: hard Vs soft singular value thresholding
        exclude_singletons: flag to exclude singleton frequency variants
        '''
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        assert λ_tv >= 0, 'λ_tv must be nonnegative'
        assert λ_r >= 0, 'λ_r must be nonnegative'
        assert 0 <= α_tv <= 1, 'α_tv must be in the unit interval'
        assert 0 <= α_r <= 1, 'α_r must be in the unit interval'
        self.bins = bins
        if bins is not None:
            bin_idxs = np.digitize(np.arange(self.n - 1), bins=bins)
            X_binned = np.zeros((len(bins), self.X.shape[1]))
            L_binned = np.zeros((len(bins), self.η.m))
            for col in range(self.X.shape[1]):
                X_binned[:, col] = np.bincount(bin_idxs,
                                               weights=self.X[:, col])
            for col in range(self.η.m):
                L_binned[:, col] = np.bincount(bin_idxs,
                                               weights=self.L[:, col])
            # stash the unbinned variables
            X_true = self.X
            L_true = self.L
            # temporarily update instance variables to the binned ones
            self.X = X_binned
            self.L = L_binned
        # badness of fit
        if exclude_singletons:
            L = self.L[1:-1, :]
            X = self.X[1:-1, :]
        else:
            L = self.L
            X = self.X
        if fit == 'prf':
            def misfit_func(Z, **kwargs):
                return -utils.ℓ(Z, X, L, **kwargs)
        elif fit == 'kl':
            def misfit_func(Z, **kwargs):
                return utils.d_kl(Z, X, L, **kwargs)
        elif fit == 'lsq':
            def misfit_func(Z, **kwargs):
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

        # Accelerated proximal gradient descent: our loss function decomposes
        # as f = g + h, where g is differentiable and h is not.
        # We transform logZ to restrict to positive solutions
        # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
        # some matrices we'll need for the first difference penalties
        D = (np.eye(self.η.m, k=0) - np.eye(self.η.m, k=-1))
        W = np.eye(self.η.m)
        W[0, 0] = 0  # W matrix deals with boundary condition
        D1 = W @ D
        D2 = D1.T @ D1

        def g(logZ, grad=False):
            '''differentiable piece of loss'''
            Z = np.exp(logZ)
            if grad:
                misfit, grad_misfit = misfit_func(Z, grad=True)
            else:
                misfit = misfit_func(Z)
            g = misfit \
                + (λ_tv / 2) * (1 - α_tv) * ((D1 @ Z) ** 2).sum() \
                + (λ_r / 2) * (1 - α_r) * (Z ** 2).sum()
            if grad:
                grad_g = grad_misfit + λ_tv * (1 - α_tv) * D2 @ Z \
                                     + λ_r * (1 - α_r) * Z
                return g, grad_g * Z
            return g

        def h(logZ):
            '''nondifferentiable piece of loss'''
            Z = np.exp(logZ)
            σ = np.linalg.svd(Z, compute_uv=False)
            if hard:
                rank_penalty = np.linalg.norm(σ, 0)
            else:
                rank_penalty = np.linalg.norm(σ, 1)
            return λ_tv * α_tv * np.abs(D1 @ Z).sum() \
                + λ_r * α_r * rank_penalty

        def f(logZ):
            '''loss'''
            return g(logZ) + h(logZ)

        # initialize using constant μ history MLE
        μ = self.constant_μ_MLE()
        logZ = np.log(μ.Z)
        # our auxiliary iterates for acceleration
        logQ = np.log(μ.Z)
        # initial loss
        f_trajectory = [f(logZ)]
        # initialize step size
        s0 = 1
        s = s0
        # max number of Armijo step size reductions
        max_line_iter = 100
        for k in range(1, max_iter + 1):
            # g(logQ) and ∇g(logQ)
            g1, grad_g1 = g(logQ, grad=True)
            # Armijo line search
            for line_iter in range(max_line_iter):
                if not np.all(np.isfinite(grad_g1)):
                    raise RuntimeError(f'invalid gradient: {grad_g1}')
                # G_s(Q) as in the notes linked above
                G = (1 / s) * (logQ - prox_update(logQ - s * grad_g1, s))
                # test g(logQ - sG_s(logQ)) for line search
                if g(logQ - s * G) <= (g1 - s * (grad_g1 * G).sum()
                                       + (s / 2) * (G ** 2).sum()):
                    break
                s *= γ
            # accelerated gradient step
            logZ_old = logZ
            logZ = logQ - s * G
            logQ = logZ + (k / (k + 3)) * (logZ - logZ_old)
            if line_iter == max_line_iter - 1:
                print('warning: line search failed')
                s = s0
            if not np.all(np.isfinite(logZ)):
                print(f'warning: Z contains invalid values')
            # terminate if loss function is constant within tolerance
            f_trajectory.append(f(logZ))
            rel_change = np.abs((f_trajectory[-1] - f_trajectory[-2])
                                / f_trajectory[-2])
            if rel_change < tol:
                print(f'relative change in loss function {rel_change:.2g} '
                      f'is within tolerance {tol} after {k} iterations')
                break
            if k == max_iter:
                print(f'maximum iteration {max_iter} reached with relative '
                      f'change in loss function {rel_change:.2g}')
        if bins is not None:
            # restore stashed unbinned variables
            self.X = X_true
            self.L = L_true
        μ.Z = np.exp(logZ)
        return μ, f_trajectory

    def plot(self, i: int = None, μ: histories.μ = None, prf_quantiles=False):
        '''plot the SFS data and optionally the expected SFS given μ

        i: component i of kSFS (default first)
        μ: mutation intensity history (optional)
        prf_quantiles: if True show 95% marginal intervals using the Poisson
                       random field
        '''
        if i is None:
            i = 0
        if μ is not None:
            z = μ.Z[:, i]
            ξ = self.L @ z
            if self.bins is not None:
                for bin in self.bins:
                    plt.axvline(bin, c='k', ls=':', alpha=0.2)
            plt.plot(range(1, self.n), ξ, 'r--', label=r'$\xi$')
            if prf_quantiles:
                ξ_lower = poisson.ppf(.025, ξ)
                ξ_upper = poisson.ppf(.975, ξ)
                plt.fill_between(range(1, self.n),
                                 ξ_lower, ξ_upper,
                                 facecolor='r', alpha=0.25,
                                 label='inner 95%\nquantile')
        plt.plot(range(1, len(self.X) + 1), self.X[:, i],
                 'k.', alpha=.25, label=r'data')
        plt.xlabel('$b$')
        plt.ylabel(r'$ξ_b$')
        plt.xscale('log')
        plt.yscale('symlog')
        plt.tight_layout()

    def clustermap(self, μ: histories.μ = None,
                   linthresh=1, **kwargs):
        '''clustermap with mixed linear-log scale color bar

        μ: inferred mutation spectrum history, χ^2 values are shown if not None
        linthresh: the range within which the plot is linear (when μ = None)
        kwargs: additional keyword arguments passed to pd.clustermap
        '''
        if μ is None:
            Z = self.X / self.X.sum(axis=1, keepdims=True)
            Z = Z / Z.mean(0, keepdims=True)
            cbar_label = 'mutation type\nenrichment'
        else:
            Ξ = self.L @ μ.Z
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
