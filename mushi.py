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

    def infer_constant(self, change_points: np.array, μ0: np.float64 = None,
                       mask: np.ndarray = None):
        u"""infer constant η and μ

        change_points: epoch change points (times)
        μ0: total mutation rate, if self.μ is None
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
                              μ0 * np.ones((len(change_points) + 1, 1)))
        t, z = μ_total.arrays()
        # number of segregating sites
        S = X_total.sum()
        # Harmonic number
        H = (1 / np.arange(1, self.n - 1)).sum()
        # constant MLE
        y = (S / 2 / H / μ0) * np.ones(len(z))

        self.η = histories.η(change_points, y)
        self.μ = histories.μ(self.η.change_points,
                             μ0 * (X.sum(axis=0, keepdims=True) / X.sum()) * np.ones((self.η.m, X.shape[1])),
                             mutation_types=self.mutation_types.values)
        self.M = utils.M(self.n, t, y)
        self.L = self.C @ self.M

    def infer_history(self,
                      α_tv: np.float64 = 0,
                      α_spline: np.float64 = 0,
                      α_ridge: np.float64 = 0,
                      β_tv: np.float64 = 0,
                      β_spline: np.float64 = 0,
                      β_spline_total: np.float64 = 0,
                      β_rank: np.float64 = 0,
                      β_ridge: np.float64 = 0,
                      max_iter: int = 1000,
                      max_line_iter=100,
                      γ: np.float64 = 0.8,
                      hard=False,
                      loss='prf',
                      mask: np.ndarray = None) -> np.ndarray:
        u"""perform sequential inference to fit η and μ

        loss parameters:
        - loss: loss function, 'prf' for Poisson random field, 'kl' for
               Kullback-Leibler divergence, 'lsq' for least-squares
        - mask: array of bools, with True indicating exclusion of that
                frequency

        η(t) regularization parameters:
        - α_tv: total variation
        - α_spline: L2 on first differences
        - α_ridge: L2 for strong convexity

        μ(t) regularization parameters:
        - β_tv: total variation
        - β_spline: L2 on first differences for each mutation type
        - β_spline_total: L2 on first differences of total mutation rate
        - β_rank: spectral regularization (soft singular value threshold)
        - β_ridge: L2 for strong convexity

        convergence parameters:
        - max_iter: maximum number of proximal gradient descent steps
        - max_line_iter: maximum number of line search steps
        - γ: step size shrinkage rate for line search
        - max_iter: maximum number of proximal gradient descent steps
        - tol: relative tolerance in objective function
        - hard: hard Vs soft singular value thresholding (l0 Vs l1 penalty)

        returns cost function
        """
        assert self.X is not None, 'use simulate() to generate data first'
        assert self.η is not None, 'must initialize e.g. with infer_constant()'

        if mask is not None:
            X = self.X[~mask, :]
        else:
            X = self.X

        t = self.η.arrays()[0]

        # badness of fit
        if loss == 'prf':
            loss = utils.prf
        elif loss == 'kl':
            loss = utils.d_kl
        elif loss == 'lsq':
            loss = utils.lsq
        else:
            raise ValueError(f'unrecognized loss argument {loss}')

        @jit
        def loss_func(logy, Z):
            L = self.C @ utils.M(self.n, t, np.exp(logy))
            if mask is not None:
                L = L[~mask, :]
            return loss(Z, X, L)

        if α_tv > 0:
            def prox_update_logy(logy, s):
                """total variation prox operator"""
                return ptv.tv1_1d(logy, s * α_tv)
        else:
            def prox_update_logy(logy, s):
                return logy

        if β_tv > 0 and β_rank > 0:
            raise NotImplementedError('fused LASSO with spectral '
                                      'regularization not available')
        elif β_tv > 0:
            def prox_update_Z(Z, s):
                """total variation prox operator on row dimension
                (oddly 1-based indexed in proxtv)
                """
                return np.maximum(ptv.tvgen(Z, [s * β_tv], [1], [1]), 0)
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
                return Z


        # Accelerated proximal gradient descent: our cost function decomposes
        # as f = g + h, where g is differentiable and h is not.
        # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

        # some matrices we'll need for the first difference penalties
        D = (np.eye(self.η.m, k=0) - np.eye(self.η.m, k=-1))
        # W matrix deals with boundary condition
        W = np.eye(self.η.m)
        W = index_update(W, index[0, 0], 0)
        D1 = W @ D  # 1st difference matrix

        @jit
        def g(logy, Z):
            """differentiable piece of cost"""
            return loss_func(logy, Z) \
                + (α_spline / 2) * ((D1 @ logy) ** 2).sum() \
                + (α_ridge / 2) * (logy ** 2).sum() \
                + (β_spline / 2) * ((D1 @ Z) ** 2).sum() \
                + (β_spline_total / 2) * ((D1 @ Z.sum(1, keepdims=True)) ** 2).sum() \
                + (β_ridge / 2) * (Z ** 2).sum()

        @jit
        def h(logy, Z):
            """nondifferentiable piece of cost"""
            σ = np.linalg.svd(Z, compute_uv=False)
            if hard:
                rank_penalty = np.linalg.norm(σ, 0)
            else:
                rank_penalty = np.linalg.norm(σ, 1)
            return α_tv * np.abs(D1 @ logy).sum() + β_tv * np.abs(D1 @ Z).sum() + β_rank * rank_penalty

        # initial iterate
        logy = np.log(self.η.y)
        Z = self.μ.Z
        # max step size
        s0 = 1

        print('inferring η(t)', flush=True)
        logy = utils.acc_prox_grad_descent(
                              logy,
                              jit(lambda logy: g(logy, Z)),
                              jit(grad(lambda logy: g(logy, Z))),
                              jit(lambda logy: h(logy, Z)),
                              prox_update_logy,
                              tol=tol,
                              max_iter=max_iter,
                              s0=s0,
                              max_line_iter=max_line_iter,
                              γ=γ)

        print('inferring μ(t) conditioned on η(t)', flush=True)
        Z = utils.three_op_prox_grad_descent(Z,
                                             jit(lambda Z: g(logy, Z)),
                                             jit(grad(lambda Z: g(logy, Z))),
                                             jit(lambda Z: h(logy, Z)),
                                             prox_update_Z,
                                             tol=tol,
                                             max_iter=max_iter,
                                             s0=s0,
                                             max_line_iter=max_line_iter,
                                             γ=γ,
                                             ls_tol=0)

        y = np.exp(logy)
        self.η = histories.η(self.η.change_points, y)
        self.M = utils.M(self.n, t, y)
        self.L = self.C @ self.M
        self.μ = histories.μ(self.η.change_points, Z,
                             mutation_types=self.mutation_types.values)
        return g(logy, Z) + h(logy, Z)

    def plot_total(self):
        """plot the total SFS"""
        x = self.X.sum(1, keepdims=True)
        plt.plot(range(1, len(x) + 1), x,
                 c='C0', ls='', marker='.', alpha=.25, label=r'data')
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

    def clustermap(self, fit: bool = False, **kwargs):
        u"""clustermap with mixed linear-log scale color bar

        fit: if True and self.μ is not None, show χ^2 fit
        kwargs: additional keyword arguments passed to pd.clustermap
        """
        if fit:
            assert self.μ is None
            Ξ = self.L @ self.μ.Z
            Z = (self.X - Ξ) ** 2 / Ξ
            cbar_label = '$\\chi^2$'
            χ2_total = Z.sum()
            p = chi2(np.prod(Z.shape)).sf(χ2_total)
            print(f'χ\N{SUPERSCRIPT TWO} goodness of fit {χ2_total}, '
                  f'p = {p}')
        else:
            Z = self.X / self.X.sum(axis=1, keepdims=True)
            Z = Z / Z.mean(0, keepdims=True)
            cbar_label = 'mutation type\nenrichment'
        df = pd.DataFrame(data=Z, index=pd.Index(range(1, self.n),
                                                 name='sample frequency'),
                          columns=self.mutation_types)
        g = sns.clustermap(df, row_cluster=False,
                           cbar_kws={'label': cbar_label}, **kwargs)
        g.ax_heatmap.set_yscale('symlog')
        return g


def main():
    """
    usage: python mushi.py -h
    """
    import argparse
    import pickle
    import configparser

    parser = argparse.ArgumentParser(description='write snps with kmer context'
                                                 ' to stdout')
    parser.add_argument('ksfs', type=str, default=None,
                        help='path to k-SFS file')
    parser.add_argument('masked_genome_size_file', type=str,
                        help='path to file containing masked genome size in nucleotides')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('outbase', type=str, default=None,
                        help='base name for output files')

    args = parser.parse_args()

    # load k-SFS
    ksfs_df = pd.read_csv(args.ksfs, sep='\t', index_col=0)
    assert np.isnan(ksfs_df.values).sum() == 0
    mutation_types = ksfs_df.columns
    n = ksfs_df.shape[0] + 1
    ksfs = kSFS(X=ksfs_df.values, mutation_types=mutation_types)

    # parse configuration file if present
    config = configparser.ConfigParser()
    config.read(args.config)

    # change points for time grid
    first = config.getfloat('change points', 'first')
    last = config['change points'].getfloat('last')
    npts = config['change points'].getint('npts')
    change_points = np.logspace(np.log10(first),
                                np.log10(last),
                                npts)

    # mask sites
    clip_low = config.getint('loss', 'clip_low', fallback=None)
    clip_high = config.getint('loss', 'clip_high', fallback=None)
    if clip_high or clip_low:
        assert clip_high is not None and clip_low is not None
        mask = np.array([False if (clip_low <= i <= n - clip_high)
                         else True
                         for i in range(n - 1)])
    else:
        mask = None

    # mutation rate estimate
    with open(args.masked_genome_size_file) as f:
        masked_genome_size = int(f.read())
    μ0 = config.getfloat('mutation rate', 'u') * masked_genome_size

    # Initialize to constant
    ksfs.infer_constant(change_points=change_points,
                        μ0=μ0,
                        mask=mask)

    tol = config.getfloat('convergence', 'tol', fallback=None)

    # parameter dict for η regularization
    η_regularization = {key: config.getfloat('η regularization', key)
                        for key in config['η regularization']}

    # parameter dict for μ regularization
    μ_regularization = {key: config.getfloat('μ regularization', key)
                        for key in config['μ regularization']
                        if key.startswith('β_')}
    if 'hard' in config['μ regularization']:
        μ_regularization['hard'] = config.getboolean('μ regularization',
                                                     'hard')

    # parameter dict for convergence parameters
    convergence = {**{key: config.getint('convergence', key)
                      for key in config['convergence']
                      if key.endswith('_iter')},
                   **{key: config.getfloat('convergence', key)
                      for key in config['convergence']
                      if not key.endswith('_iter')}}

    # parameter dict for loss parameters
    loss = dict(mask=mask)
    if 'fit' in config['loss']:
        loss['fit'] = config.get('loss', 'fit')

    print('sequential inference of η(t) and μ(t)\n', flush=True)
    f = ksfs.infer_history(**loss, **η_regularization, **μ_regularization,
                           **convergence)

    plt.figure(figsize=(6, 9))
    plt.subplot(321)
    ksfs.plot_total()
    plt.subplot(322)
    ksfs.η.plot()
    plt.subplot(323)
    ksfs.plot(normed=True, alpha=0.5, rasterized=True)
    plt.subplot(324)
    ksfs.μ.plot_cumulative(rasterized=True)
    plt.subplot(325)
    plt.plot(ksfs.η.change_points, ksfs.tmrca_cdf())
    plt.xlabel('$t$')
    plt.ylabel('TMRCA CDF')
    plt.ylim([0, 1])
    plt.xscale('symlog')
    plt.tight_layout()
    plt.subplot(326)
    plt.plot(range(1, 1 + min(ksfs.μ.Z.shape)),
             np.linalg.svd(ksfs.μ.Z, compute_uv=False), '.')
    plt.xlabel('singular value rank')
    plt.ylabel('singular value')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{args.outbase}.fit.pdf')

    # pickle the final ksfs (which contains all the inferred history info)
    with open(f'{args.outbase}.pkl', 'wb') as f:
        pickle.dump(ksfs, f)


if __name__ == '__main__':
    main()
