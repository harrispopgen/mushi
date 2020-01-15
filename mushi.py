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

from typing import List, Dict

import histories
import utils

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import composition as cmp

class kSFS():
    """The kSFS model described in the text"""

    def __init__(self, X: np.ndarray = None, n: int = None,
                 mutation_types: List[str] = None):
        u"""Sample frequency spectrum

        X: observed k-SFS matrix (optional)
        n: number of haplotypes (optional)
        mutation_types: names of X columns
        """
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
            raise ValueError('either X or n must be specified')
        else:
            self.n = n
        self.C = utils.C(self.n)
        self.η = None
        self.μ = None
        self.M = None
        self.L = None

    def tmrca_cdf(self, η: histories.η) -> onp.ndarray:
        """The CDF of the TMRCA of at each change point

        η: demographic history
        """
        if η is None:
            raise ValueError('η(t) must be defined first')
        t, y = η.arrays()
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

    def simulate(self, η: histories.η, μ: histories.μ,
                 seed: int = None) -> None:
        """simulate a SFS under the Poisson random field model (no linkage)
        assigns simulated SFS to self.X

        η: demographic history
        μ: mush
        seed: random seed (optional)
        """
        if not η.check_grid(μ):
            raise ValueError('η and μ histories must use the same time grid')
        onp.random.seed(seed)
        t, y = η.arrays()
        M = utils.M(self.n, t, y)
        L = self.C @ M

        self.X = poisson.rvs(L @ μ.Z)
        self.mutation_types = μ.mutation_types

    def infer_history(self,
                      change_points: np.array,
                      μ0: np.float64,
                      η: histories.η = None,
                      infer_η: bool = True,
                      infer_μ: bool = True,
                      α_tv: np.float64 = 0,
                      α_spline: np.float64 = 0,
                      α_ridge: np.float64 = 0,
                      β_tv: np.float64 = 0,
                      β_spline: np.float64 = 0,
                      β_rank: np.float64 = 0,
                      hard: bool = False,
                      β_ridge: np.float64 = 0,
                      max_iter: int = 1000,
                      s0: int = 1,
                      max_line_iter=100,
                      γ: np.float64 = 0.8,
                      tol: np.float64 = 1e-4,
                      loss='prf',
                      mask: np.ndarray = None) -> Dict:
        u"""perform sequential inference to fit η and μ

        change_points: epoch change points (times)
        μ0: total mutation rate (per genome per generation)
        η: optional demographic history. If None (the default), it will be
           inferred from the total SFS

        infer_η, infer_μ: flags can be set to False to skip either optimization

        loss parameters:
        - loss: loss function, 'prf' for Poisson random field, 'kl' for
               Kullback-Leibler divergence, 'lsq' for least-squares
        - mask: array of bools, with False indicating exclusion of that
                frequency

        η(t) regularization parameters:
        - α_tv: total variation
        - α_spline: L2 on first differences
        - α_ridge: L2 for strong convexity

        μ(t) regularization parameters:
        - hard: hard singular value thresholding (non-convex)
        - β_tv: total variation
        - β_spline: L2 on first differences for each mutation type
        - β_rank: spectral regularization (soft singular value threshold)
        - β_ridge: L2 for strong convexity

        convergence parameters:
        - max_iter: maximum number of proximal gradient descent steps
        - tol: relative tolerance in objective function
        - s0: max step size
        - max_line_iter: maximum number of line search steps
        - γ: step size shrinkage rate for line search

        return: a dictionary of inference metadata
        """

        metadata = {}

        assert self.X is not None, 'use simulate() to generate data first'
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        if mask is not None:
            assert len(mask) == self.X.shape[0], 'mask must have n-1 elements'

        # ininitialize with MLE constant η and μ
        self.X
        x = self.X.sum(1, keepdims=True)
        μ_total = histories.μ(change_points,
                              μ0 * np.ones((len(change_points) + 1, 1)))
        t, z = μ_total.arrays()
        # number of segregating variants in each mutation type
        S = self.X.sum(0, keepdims=True)

        if η is not None:
            self.η = η
        elif self.η is None:
            # Harmonic number
            H = (1 / np.arange(1, self.n - 1)).sum()
            # constant MLE
            y = (S.sum() / 2 / H / μ0) * np.ones(len(z))
            self.η = histories.η(change_points, y)
        # NOTE: scaling by S is a hack, should use relative triplet
        #       content of masked genome
        if self.μ is None:
            self.μ = histories.μ(self.η.change_points,
                                 μ0 * (S / S.sum()) * np.ones((self.η.m,
                                                               self.X.shape[1])),
                                 mutation_types=self.mutation_types.values)
        self.M = utils.M(self.n, t, self.η.y)
        self.L = self.C @ self.M

        # badness of fit
        if loss == 'prf':
            def loss(*args, **kwargs):
                return -utils.prf(*args, **kwargs)
        elif loss == 'kl':
            loss = utils.d_kl
        elif loss == 'lsq':
            loss = utils.lsq
        else:
            raise ValueError(f'unrecognized loss argument {loss}')

        # some matrices we'll need for the first difference penalties
        D = (np.eye(self.η.m, k=0) - np.eye(self.η.m, k=-1))
        # W matrix deals with boundary condition
        W = np.eye(self.η.m)
        W = index_update(W, index[0, 0], 0)
        D1 = W @ D  # 1st difference matrix
        # D2 = D1.T @ D1  # 2nd difference matrix

        if η is None and infer_η:
            print('inferring η(t)', flush=True)

            # Accelerated proximal gradient descent: our objective function
            # decomposes as f = g + h, where g is differentiable and h is not.
            # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

            @jit
            def g(logy):
                """differentiable piece of objective in η problem"""
                L = self.C @ utils.M(self.n, t, np.exp(logy))
                if mask is not None:
                    loss_term = loss(z, x[mask, :], L[mask, :])
                else:
                    loss_term = loss(z, x, L)
                return loss_term + (α_spline / 2) * ((D1 @ logy) ** 2).sum() \
                                 + (α_ridge / 2) * (logy ** 2).sum()

            if α_tv > 0:
                @jit
                def h(logy):
                    """nondifferentiable piece of objective in η problem"""
                    return α_tv * np.abs(D1 @ logy).sum()

                def prox(logy, s):
                    """total variation prox operator"""
                    return ptv.tv1_1d(logy, s * α_tv)
            else:
                @jit
                def h(logy):
                    return 0

                def prox(logy, s):
                    return logy

            # initial iterate
            logy = np.log(self.η.y)

            logy, convergence = utils.acc_prox_grad_descent(logy, g, jit(grad(g)), h,
                                               prox,
                                               tol=tol,
                                               max_iter=max_iter,
                                               s0=s0,
                                               max_line_iter=max_line_iter,
                                               γ=γ)

            metadata['y_convergence'] = convergence

            y = np.exp(logy)

            self.η = histories.η(self.η.change_points, y)
            self.M = utils.M(self.n, t, y)
            self.L = self.C @ self.M

        if infer_μ:
            print('inferring μ(t) conditioned on η(t)', flush=True)

            # orthonormal basis for Aitchison simplex
            # NOTE: instead of Gram-Schmidt could try SVD of clr transformed X
            #       https://en.wikipedia.org/wiki/Compositional_data#Isometric_logratio_transform
            basis = cmp._gram_schmidt_basis(self.μ.Z.shape[1])
            # initial iterate in inverse log-ratio transform
            Z = cmp.ilr(self.μ.Z, basis)

            @jit
            def g(Z):
                """differentiable piece of objective in μ problem"""
                if mask is not None:
                    loss_term = loss(μ0 * cmp.ilr_inv(Z, basis), self.X[mask, :],
                                     self.L[mask, :])
                else:
                    loss_term = loss(μ0 * cmp.ilr_inv(Z, basis), self.X, self.L)
                return loss_term + (β_spline / 2) * ((D1 @ Z) ** 2).sum() \
                                 + (β_ridge / 2) * (Z ** 2).sum()

            if β_tv and β_rank:
                @jit
                def h1(Z):
                    """1st nondifferentiable piece of objective in μ problem"""
                    return β_tv * np.abs(D1 @ Z).sum()

                def prox1(Z, s):
                    """total variation prox operator on row dimension
                    (oddly 1-based indexed in proxtv)
                    """
                    return ptv.tvgen(Z, [s * β_tv], [1], [1])

                @jit
                def h2(Z):
                    """2nd nondifferentiable piece of objective in μ problem"""
                    σ = np.linalg.svd(Z, compute_uv=False)
                    return β_rank * np.linalg.norm(σ, 0 if hard else 1)

                def prox2(Z, s):
                    """singular value thresholding"""
                    U, σ, Vt = np.linalg.svd(Z, full_matrices=False)
                    if hard:
                        σ = index_update(σ, index[σ <= s * β_rank], 0)
                    else:
                        σ = np.maximum(0, σ - s * β_rank)
                    Σ = np.diag(σ)
                    return U @ Σ @ Vt

                Z, convergence = utils.three_op_prox_grad_descent(Z, g, jit(grad(g)), h1, prox1,
                                                     h2, prox2,
                                                     tol=tol,
                                                     max_iter=max_iter,
                                                     s0=s0,
                                                     max_line_iter=max_line_iter,
                                                     γ=γ, ls_tol=0)

                metadata['Z_convergence'] = convergence

            else:
                if β_tv:
                    @jit
                    def h(Z):
                        """nondifferentiable piece of objective in μ problem"""
                        return β_tv * np.abs(D1 @ Z).sum()

                    def prox(Z, s):
                        """total variation prox operator on row dimension
                        (oddly 1-based indexed in proxtv)
                        """
                        return ptv.tvgen(Z, [s * β_tv], [1], [1])
                elif β_rank:
                    @jit
                    def h(Z):
                        """nondifferentiable piece of objective in μ problem"""
                        σ = np.linalg.svd(Z, compute_uv=False)
                        return β_rank * np.linalg.norm(σ, 0 if hard else 1)

                    def prox(Z, s):
                        """singular value thresholding"""
                        U, σ, Vt = np.linalg.svd(Z, full_matrices=False)
                        if hard:
                            σ = index_update(σ, index[σ <= s * β_rank], 0)
                        else:
                            σ = np.maximum(0, σ - s * β_rank)
                        Σ = np.diag(σ)
                        return U @ Σ @ Vt
                else:
                    @jit
                    def h(Z):
                        return 0

                    @jit
                    def prox(Z, s):
                        return Z

                Z, convergence = utils.acc_prox_grad_descent(Z, g, jit(grad(g)), h, prox,
                                                tol=tol,
                                                max_iter=max_iter,
                                                s0=s0,
                                                max_line_iter=max_line_iter,
                                                γ=γ)

                metadata['Z_convergence'] = convergence

            self.μ = histories.μ(self.η.change_points,
                                 μ0 * cmp.ilr_inv(Z, basis),
                                 mutation_types=self.mutation_types.values)

        return metadata

    def plot_total(self, **kwargs):
        """plot the total SFS

        kwargs: key word arguments passed to plt.plot (e.g. color)
        """
        x = self.X.sum(1, keepdims=True)
        plt.plot(range(1, len(x) + 1), x, ls='', marker='.', alpha=.25,
                 **kwargs)
        if self.η is not None:
            if 'label' in kwargs:
                del kwargs['label']
            if self.μ is not None:
                z = self.μ.Z.sum(1)
            else:
                z = np.ones_like(self.η.y)
            ξ = self.L.dot(z)
            plt.plot(range(1, self.n), ξ, ls='--', **kwargs)
            ξ_lower = poisson.ppf(.025, ξ)
            ξ_upper = poisson.ppf(.975, ξ)
            if 'c' in kwargs:
                kwargs['facecolor'] = kwargs.pop('c')
            if 'color' in kwargs:
                kwargs['facecolor'] = kwargs.pop('color')
            plt.fill_between(range(1, self.n),
                             ξ_lower, ξ_upper, alpha=0.25, **kwargs)
        plt.xlabel('sample frequency')
        plt.ylabel(r'variant count')
        plt.xscale('log')
        plt.yscale('symlog')
        plt.tight_layout()

    def plot(self, type=None, clr: bool = False, **kwargs) -> None:
        """
        clr: flag to normalize to total mutation intensity and display as
             centered log ratio transform
        kwargs: key word arguments passed to plt.plot
        """
        if self.μ is not None:
            Ξ = self.L @ self.μ.Z
        if clr:
            X = cmp.clr(self.X)
            if self.μ is not None:
                Ξ = cmp.clr(Ξ)
            plt.ylabel('variant count composition\n(CLR transformed)')
        else:
            X = self.X
            plt.ylabel('number of variants')
        if type is not None:
            i = self.mutation_types.get_loc(type)
            X = X[:, i]
            if self.μ is not None:
                Ξ = Ξ[:, i]

        if self.μ is not None:
            plt.plot(range(1, self.n), X, ls='', marker='.', rasterized=True,
                     **kwargs)
            line_kwargs = kwargs
            if 'label' in line_kwargs:
                del line_kwargs['label']
            plt.gca().set_prop_cycle(None)
            plt.plot(range(1, self.n), Ξ, **line_kwargs)
        else:
            plt.plot(range(1, self.n), X, **kwargs)
        plt.xlabel('sample frequency')
        plt.xscale('log')
        if 'label' in kwargs:
            plt.legend()
        plt.tight_layout()

    def clustermap(self, **kwargs):
        u"""clustermap of compositionally centralized k-SFS

        kwargs: additional keyword arguments passed to pd.clustermap
        """
        Z = cmp.centralize(self.X)
        cbar_label = 'variant count\nperturbation'
        df = pd.DataFrame(data=Z, index=pd.Index(range(1, self.n),
                                                 name='sample frequency'),
                          columns=self.mutation_types)
        g = sns.clustermap(df, row_cluster=False,
                           center=1 / Z.shape[1],
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
        mask = np.array([True if (clip_low <= i < n - clip_high - 1)
                         else False
                         for i in range(n - 1)])
    else:
        mask = None

    # mutation rate estimate
    with open(args.masked_genome_size_file) as f:
        masked_genome_size = int(f.read())
    μ0 = config.getfloat('population', 'u') * masked_genome_size

    # generation time
    t_gen = config.getfloat('population', 't_gen', fallback=None)

    # parameter dict for η regularization
    η_regularization = {key: config.getfloat('η regularization', key)
                        for key in config['η regularization']}

    # parameter dict for μ regularization
    μ_regularization = {key: config.getfloat('μ regularization', key)
                        for key in config['μ regularization']
                        if key.startswith('β_')}
    if 'hard' in config['μ regularization']:
        μ_regularization['hard'] = config.getboolean('μ regularization', 'hard')

    # parameter dict for convergence parameters
    convergence = {key: config.getint('convergence', key)
                   if key.endswith('_iter')
                   else config.getfloat('convergence', key)
                   for key in config['convergence']}
    # parameter dict for loss parameters
    loss = dict(mask=mask)
    if 'loss' in config['loss']:
        loss['loss'] = config.get('loss', 'loss')

    print('sequential inference of η(t) and μ(t)\n', flush=True)
    ksfs.infer_history(change_points, μ0, **loss, **η_regularization,
                       **μ_regularization, **convergence)

    plt.figure(figsize=(7, 9))
    plt.subplot(321)
    ksfs.plot_total()
    plt.subplot(322)
    ksfs.η.plot(t_gen=t_gen,
                # ds='steps-post'
                )
    plt.subplot(323)
    ksfs.plot(clr=True, alpha=0.5)
    plt.subplot(324)
    ksfs.μ.plot(t_gen=t_gen, clr=True, alpha=0.5)
    plt.subplot(325)
    if t_gen:
        plt.plot(t_gen * ksfs.η.change_points, ksfs.tmrca_cdf(ksfs.η))
        plt.xlabel('$t$ (years ago)')
    else:
        plt.plot(ksfs.η.change_points, ksfs.tmrca_cdf(ksfs.η))
        plt.xlabel('$t$ (generations ago)')
    plt.ylabel('TMRCA CDF')
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.subplot(326)
    Z = cmp.clr(ksfs.μ.Z)
    plt.plot(range(1, 1 + min(Z.shape)),
             np.linalg.svd(Z, compute_uv=False), '.')
    plt.xlabel('singular value rank')
    plt.xscale('log')
    plt.ylabel('singular value')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{args.outbase}.fit.pdf')

    # pickle the final ksfs (which contains all the inferred history info)
    with open(f'{args.outbase}.pkl', 'wb') as f:
        pickle.dump(ksfs, f)


if __name__ == '__main__':
    main()
