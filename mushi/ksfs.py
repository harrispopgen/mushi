#! /usr/bin/env python

import mushi.histories as hst
from mushi import loss_functions, utils
import mushi.optimization as opt
import mushi.composition as cmp

from jax.config import config
import numpy as onp
import jax.numpy as np
from jax import jit, grad
from jax.scipy.special import expit, logit
from scipy.stats import poisson
from typing import Union, List, Dict, Tuple
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

config.update('jax_enable_x64', True)
# config.update('jax_debug_nans', True)

class kSFS():
    r"""The core :math:`k`-SFS class for simulation and inference

    Attributes:
        X (:obj:`numpy.ndarray`): :math:`k`-SFS matrix (or 1D SFS vector)
        eta (:obj:`mushi.eta`): demographic history
        mu (:obj:`mushi.mu`): mutation spectrum history
        mutation_types (:obj:`List[str]`): mutation spectrum history
        n (:obj:`int`): number of sampled haplotypes

    Notes:
        Three constructors:

        1) ``ksfs_file``: path to k-SFS file, as ouput by `mutyper ksfs`

        2) ``X``: k-SFS matrix

        3) ``n``: number of haplotypes to initialize for simulation

    Args:
        ksfs_file: path to :math:`k`-SFS file, as ouput by ``mutyper ksfs``
        X: :math:`k`-SFS matrix
        n: number of sampled haplotypes
        mutation_types: list of names of X columns
                        (defaults to integer range)

    """

    def __init__(self, X: np.ndarray = None, mutation_types: List[str] = None,
                 file: str = None,
                 n: int = None):
        if file is not None:
            df = pd.read_csv(file, sep='\t', index_col=0)
            assert np.all(df.values >= 0)
            n = df.shape[0] + 1
            self.X = df.values
            self.n = len(self.X) + 1
            self.mutation_types = pd.Index(df.columns,
                                           name='mutation type')

        elif X is not None:
            # if 1D SFS, make a column vector
            if X.ndim == 1:
                X = X[:, np.newaxis]
            self.X = X
            self.n = len(X) + 1
            if mutation_types is not None:
                self.mutation_types = pd.Index(mutation_types,
                                               name='mutation type')
            else:
                self.mutation_types = pd.Index(range(self.X.shape[1]),
                                               name='mutation type')
        elif n is None:
            raise TypeError('either file, or X, or n must be specified')
        else:
            self.n = n
            self.X = None
        self.C = utils.C(self.n)
        self.η = None
        self.μ = None
        self.M = None
        self.L = None

        # ancestral state misidentification
        # misidentification rate
        self.r = None
        # frequency misidentification operator
        self.AM_freq = np.eye(self.n - 1)[::-1]
        # mutation type misidentification operator
        if self.X is not None and self.X.shape[1] > 1:
            self.AM_mut = utils.mutype_misid(self.mutation_types)

    @property
    def eta(self) -> hst.eta:
        r"""Read-only alias to η attribute
        """
        return self.η

    def check_eta(self):
        if self.η is None:
            raise TypeError('demographic history η is not defined')

    @property
    def mu(self) -> hst.mu:
        r"""Read-only alias to μ attribute
        """
        return self.μ

    def as_df(self) -> pd.DataFrame:
        r"""Return a pandas DataFrame representation
        """
        return pd.DataFrame(self.X, index=range(1, self.n),
                            columns=self.mutation_types)

    def clear_eta(self) -> None:
        r"""Clear demographic history attribute η
        """
        self.η = None
        self.M = None
        self.L = None

    def clear_mu(self) -> None:
        r"""Clear μ attribute
        """
        self.μ = None

    def tmrca_cdf(self, eta: hst.eta = None) -> onp.ndarray:
        r"""The CDF of the TMRCA at each change point in ``eta``

        Args:
            eta: demographic history (if ``None``, use ``self.eta`` attribute)
        """
        if eta is None:
            self.check_eta()
            eta = self.η
        t, y = eta.arrays()
        return 1 - utils.tmrca_sf(t, y, self.n)[1:-1]

    def simulate(self, eta: hst.eta, mu: Union[hst.mu, np.float64],
                 r: np.float64 = 0,
                 seed: int = None) -> None:
        r"""Simulate a :math:`k`-SFS under the Poisson random field model
        (no linkage), assign to ``X`` attribute

        Args:
            eta: demographic history
            mu: mutation spectrum history (or constant mutation rate)
            r: ancestral state misidentification rate (default 0)
            seed: random seed
        """
        onp.random.seed(seed)
        t, y = eta.arrays()
        M = utils.M(self.n, t, y)
        L = self.C @ M
        if type(mu) == hst.mu:
            eta.check_grid(mu)
        else:
            mu = hst.mu(eta.change_points, mu * np.ones_like(y))
        Ξ = L @ mu.Z
        self.mutation_types = mu.mutation_types
        if len(self.mutation_types) == 1:
            self.AM_mut = np.array([[1]])
        else:
            self.AM_mut = utils.mutype_misid(self.mutation_types)
        self.X = poisson.rvs((1 - r) * Ξ + r * self.AM_freq @ Ξ @ self.AM_mut)

    def infer_eta(self,
                  mu0: np.float64,
                  folded: bool = False,
                  trend_penalties: List[Tuple[int, np.float64]] = [],
                  ridge_penalty: np.float64 = 0,
                  pts: np.float64 = 100,
                  ta: np.float64 = None,
                  log_transform: bool = True,
                  eta: hst.eta = None,
                  eta_ref: hst.eta = None,
                  loss: str = 'prf',
                  max_iter: int = 1000,
                  tol: np.float64 = 0,
                  line_search_kwargs: Dict = {},
                  trend_kwargs: Dict = {},
                  verbose: bool = False
                  ) -> None:
        r"""infer demographic history :math:`\eta(t)`

        Args:
            mu0: total mutation rate (per genome per generation)
            folded: if ``False``, infer :math:`\eta(t)` using unfolded SFS. If
                    ``True``, can only be used with ``infer_mu=False``, and
                    infer :math:`\eta(t)` using folded SFS.
            trend_penalties: list of tuples (k, λ) for kth order trend penalties
            ridge_penalty: ridge penalty
            pts: number of points for time discretization
            ta: time (in WF generations ago) of oldest change point in time
                discretization. If ``None``, set automatically based on
                10 * E[TMRCA] under MLE constant demography
            log_transform: fit :math:`\log\eta(t)`
            eta: initial demographic history. By default, a constant MLE is
                 computed
            eta_ref: reference demographic history for ridge penalty. If
                     ``None``, the constant MLE is used
            loss: loss function name from loss_functions module
            max_iter: maximum number of optimization steps
            tol: relative tolerance in objective function (if ``0``, not used)
            line_search_kwargs: line search keyword arguments,
                                see :py:meth:`mushi.optimization.LineSearcher`
            trend_kwargs: keyword arguments for trend filtering,
                          see :py:meth:`mushi.optimization.TrendFilter.run`
            verbose: print verbose messages if ``True``
        """
        if self.X is None:
            raise TypeError('use simulate() to generate data first')

        # total SFS
        x = self.X.sum(1)
        # fold the spectrum if inference is on folded SFS
        if folded:
            x = utils.fold(x)

        # constant MLE
        # Harmonic number
        H = (1 / np.arange(1, self.n - 1)).sum()
        N_const = (self.X.sum() / 2 / H / mu0)

        if ta is None:
            tmrca_exp = 4 * N_const * (1 - 1 / self.n)
            ta = 10 * tmrca_exp
        change_points = np.logspace(0, np.log10(ta), pts)

        # ininitialize with MLE constant η
        if eta is not None:
            self.η = eta
        elif self.η is None:
            y = N_const * np.ones(change_points.size + 1)
            self.η = hst.eta(change_points, y)
        t = self.η.arrays()[0]

        self.mu0 = mu0
        self.M = utils.M(self.n, t, self.η.y)
        self.L = self.C @ self.M

        # badness of fit
        loss = getattr(loss_functions, loss)

        # Accelerated proximal gradient method: our objective function
        # decomposes as f = g + h, where g is differentiable and h is not.
        # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

        # Tikhonov matrix
        if eta_ref is None:
            eta_ref = self.η
            Γ = np.diag(np.ones_like(eta_ref.y))
        else:
            # - log(1 - CDF)
            Γ = np.diag(-np.log(utils.tmrca_sf(t, eta_ref.y, self.n))[:-1])
        y_ref = np.log(eta_ref.y) if log_transform else eta_ref.y

        # In the following, the parameter vector params will contain the
        # misid rate in params[0], and y in params[1:]
        @jit
        def g(params):
            """differentiable piece of objective in η problem"""
            y = params[1:]
            if log_transform:
                M = utils.M(self.n, t, np.exp(y))
            else:
                M = utils.M(self.n, t, y)
            L = self.C @ M
            ξ = self.mu0 * L.sum(1)
            if folded:
                ξ = utils.fold(ξ)
            else:
                r = expit(params[0])
                ξ = (1 - r) * ξ + r * self.AM_freq @ ξ
            loss_term = loss(np.squeeze(ξ), x)
            y_delta = y - y_ref
            ridge_term = (ridge_penalty / 2) * (y_delta.T @ Γ @ y_delta)
            return loss_term + ridge_term

        @jit
        def h(params):
            """nondifferentiable piece of objective in η problem"""
            return sum(λ * np.linalg.norm(np.diff(params[1:], k), 1)
                       for k, λ in trend_penalties)

        def prox(params, s):
            """trend filtering prox operator (no jit due to ptv module)"""
            if trend_penalties:
                k, sλ = zip(*((k, s * λ) for k, λ in trend_penalties))
                trend_filterer = opt.TrendFilter(k, sλ)
                params = params.at[1:].set(trend_filterer.run(params[1:],
                                                              **trend_kwargs))
            if log_transform:
                return params
            # else:
            # clip to minimum population size of 1
            return np.clip(params, 1)

        # optimizer
        optimizer = opt.AccProxGrad(g, jit(grad(g)), h, prox, verbose=verbose, **line_search_kwargs)
        # initial point
        params = np.concatenate((np.array([logit(1e-3)]),
                                 np.log(self.η.y) if log_transform else self.η.y))
        # run optimization
        params = optimizer.run(params, tol=tol, max_iter=max_iter)

        if not folded:
            self.r = expit(params[0])
        y = np.exp(params[1:]) if log_transform else params[1:]
        self.η = hst.eta(self.η.change_points, y)
        self.M = utils.M(self.n, t, y)
        self.L = self.C @ self.M

    def infer_mush(self,
                   trend_penalties: List[Tuple[int, np.float64]] = [],
                   ridge_penalty: np.float64 = 0,
                   rank_penalty: np.float64 = 0,
                   hard: bool = False,
                   mu_ref: hst.mu = None,
                   loss: str = 'prf',
                   max_iter: int = 1000,
                   tol: np.float64 = 0,
                   line_search_kwargs: Dict = {},
                   trend_kwargs: Dict = {},
                   verbose: bool = False
                   ) -> None:
        r"""Infer mutation spectrum history :math:`\mu(t)`

        Args:
            trend_penalties: list of tuples (k, λ) for kth order trend penalties
            ridge_penalty: ridge penalty
            rank_penalty: rank penalty
            hard: hard rank penalty (non-convex)
            mu_ref: reference MuSH for ridge penalty. If None, the constant
                    MLE is used
            loss: loss function from loss_functions module
            max_iter: maximum number of optimization steps
            tol: relative tolerance in objective function (if ``0``, not used)
            line_search_kwargs: line search keyword arguments,
                                see :py:class:`mushi.optimization.LineSearcher`
            trend_kwargs: keyword arguments for trend filtering,
                          see :py:meth:`mushi.optimization.TrendFilter.run`
            verbose: print verbose messages if ``True``
        """
        if self.X is None:
            raise TypeError('use simulate() to generate data first')
        self.check_eta()
        if len(self.mutation_types) < 2:
            raise ValueError('k-SFS must contain multiple mutation types')

        # number of segregating variants in each mutation type
        S = self.X.sum(0, keepdims=True)
        # ininitialize with MLE constant μ
        μ_const = hst.mu(self.η.change_points,
                         self.mu0 * (S / S.sum()) * np.ones((self.η.m,
                                                        self.X.shape[1])),
                         mutation_types=self.mutation_types.values)
        if self.μ is None:
            self.μ = μ_const
        t = μ_const.arrays()[0]
        self.M = utils.M(self.n, t, self.η.y)
        self.L = self.C @ self.M

        # badness of fit
        loss = getattr(loss_functions, loss)

        if mu_ref is None:
            mu_ref = μ_const
            # Tikhonov matrix
            Γ = np.diag(np.ones_like(self.η.y))
        else:
            # - log(1 - CDF)
            Γ = np.diag(- np.log(utils.tmrca_sf(t, self.η.y, self.n))[:-1])

        # orthonormal basis for Aitchison simplex
        # NOTE: instead of Gram-Schmidt could try SVD of clr transformed X
        #       https://en.wikipedia.org/wiki/Compositional_data#Isometric_logratio_transform
        basis = cmp._gram_schmidt_basis(self.μ.Z.shape[1])
        # initial iterate in inverse log-ratio transform
        Z = cmp.ilr(self.μ.Z, basis)
        Z_const = cmp.ilr(μ_const.Z, basis)
        Z_ref = cmp.ilr(mu_ref.Z, basis)

        @jit
        def g(Z):
            """differentiable piece of objective in μ problem"""
            Ξ = self.L @ (self.mu0 * cmp.ilr_inv(Z, basis))
            if self.r is not None:
                Ξ = (1 - self.r) * Ξ + self.r * self.AM_freq @ Ξ @ self.AM_mut
            loss_term = loss(Ξ, self.X)
            Z_delta = Z - Z_ref
            ridge_term = (ridge_penalty / 2) * np.sum(Z_delta * (Γ @ Z_delta))
            return loss_term + ridge_term

        @jit
        def h_trend(Z):
            """trend filtering penalty"""
            return sum(λ * np.linalg.norm(np.diff(Z, k, axis=0), 1)
                       for k, λ in trend_penalties)

        def prox_trend(Z, s):
            """trend filtering prox operator (no jit due to ptv module)"""
            k, sλ = zip(*((k, s * λ) for k, λ in trend_penalties))
            trend_filterer = opt.TrendFilter(k, sλ)
            return trend_filterer.run(Z, **trend_kwargs)

        @jit
        def h_rank(Z):
            """2nd nondifferentiable piece of objective in μ problem"""
            if hard:
                return rank_penalty * np.linalg.matrix_rank(Z - Z_const)
            # else:
            return rank_penalty * np.linalg.norm(Z - Z_const, 'nuc')

        @jit
        def prox_rank(Z, s):
            """singular value thresholding"""
            U, σ, Vt = np.linalg.svd(Z - Z_const, full_matrices=False)
            if hard:
                σ = σ.at[σ <= s * β_rank].set(0)
            else:
                σ = np.maximum(0, σ - s * rank_penalty)
            Σ = np.diag(σ)
            return Z_const + U @ Σ @ Vt

        # optimizer
        if trend_penalties and rank_penalty:
            optimizer = opt.ThreeOpProxGrad(g, jit(grad(g)),
                                            h_trend, prox_trend,
                                            h_rank, prox_rank,
                                            verbose=verbose,
                                            **line_search_kwargs)
        else:
            if trend_penalties:
                h = h_trend
                prox = prox_trend
            elif rank_penalty:
                h = h_trend
                prox = prox_trend
            else:
                @jit
                def h(Z):
                    return 0

                @jit
                def prox(Z, s):
                    return Z

            optimizer = opt.AccProxGrad(g, jit(grad(g)), h, prox,
                                        verbose=verbose, **line_search_kwargs)

        # run optimizer
        Z = optimizer.run(Z, tol=tol, max_iter=max_iter)

        self.μ = hst.mu(self.η.change_points,
                        self.mu0 * cmp.ilr_inv(Z, basis),
                        mutation_types=self.mutation_types.values)

    def plot_total(self, kwargs: Dict = dict(ls='', marker='.'),
                   line_kwargs: Dict = dict(),
                   fill_kwargs: Dict = dict(),
                   folded: bool = False) -> None:
        r"""Plot the SFS using matplotlib

        Args:
            kwargs: keyword arguments for scatter plot
            line_kwargs: keyword arguments for expectation line
            fill_kwargs: keyword arguments for marginal fill
            folded: if ``True``, plot the folded SFS and fit
        """
        x = self.X.sum(1, keepdims=True)
        if folded:
            x = utils.fold(x)
        plt.plot(range(1, len(x) + 1), x, **kwargs)
        if self.η is not None:
            if 'label' in kwargs:
                del kwargs['label']
            if self.μ is not None:
                self.η.check_grid(self.μ)
                z = self.μ.Z.sum(1)
            else:
                z = self.mu0 * np.ones_like(self.η.y)
            ξ = self.L.dot(z)
            if folded:
                ξ = utils.fold(onp.array(ξ))
            else:
                if self.r is None:
                    raise TypeError('ancestral state misidentification rate '
                                    'is not inferred, do you want '
                                    'folded=True?')
                ξ = (1 - self.r) * ξ + self.r * self.AM_freq @ ξ
            plt.plot(range(1, len(ξ) + 1), ξ, **line_kwargs)
            ξ_lower = poisson.ppf(.025, ξ)
            ξ_upper = poisson.ppf(.975, ξ)
            plt.fill_between(range(1, len(ξ) + 1),
                             ξ_lower, ξ_upper, **fill_kwargs)
        plt.xlabel('sample frequency')
        plt.ylabel(r'variant count')
        plt.tight_layout()

    def plot(self, types=None, clr: bool = False,
             kwargs: Dict = dict(ls='', marker='.', rasterized=True),
             line_kwargs: Dict = dict()) -> None:
        r"""Plot the :math:`k`-SFS

        Args:
            types: iterable of mutation type names to restrict plotting to
            clr: flag to normalize to total mutation intensity and display as
                 centered log ratio transform
            kwargs: key word arguments passed to data scatter plot
            line_kwargs: key word arguments passed to expectation line plot
        """
        if self.μ is not None:
            Ξ = self.L @ self.μ.Z
            Ξ = (1 - self.r) * Ξ + self.r * self.AM_freq @ Ξ @ self.AM_mut
        if clr:
            X = cmp.clr(self.X)
            if self.μ is not None:
                Ξ = cmp.clr(Ξ)
            plt.ylabel('variant count composition\n(CLR transformed)')
        else:
            X = self.X
            plt.ylabel('number of variants')
        if types is not None:
            idxs = [self.mutation_types.get_loc(type) for type in types]
            X = X[:, idxs]
            if self.μ is not None:
                Ξ = Ξ[:, idxs]

        plt.plot(range(1, self.n), X, **kwargs)
        if self.μ is not None:
            plt.gca().set_prop_cycle(None)
            plt.plot(range(1, self.n), Ξ, **line_kwargs)
        plt.xlabel('sample frequency')
        plt.xscale('log')
        plt.tight_layout()

    def clustermap(self, **kwargs) -> None:
        r"""Clustermap of compositionally centralized k-SFS

        Args:
            kwargs: additional keyword arguments passed to pandas.clustermap
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
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(),
                                     fontsize=9, family='monospace')

    def loss(self, func='prf') -> onp.float64:
        """Loss under current history

        Args:
            func: loss function name from loss_functions module

        Returns:
            loss
        """
        self.check_eta()
        loss = getattr(loss_functions, func)
        if self.μ is None:
            ξ = self.mu0 * self.L.sum(1)
            if folded:
                ξ = utils.fold(ξ)
                x = utils.fold(X.sum(1))
            else:
                ξ = (1 - self.r) * ξ + self.r * self.AM_freq @ ξ
            return loss(np.squeeze(ξ), np.squeeze(x))
        # else:
        Ξ = self.L @ self.μ.Z
        if self.r is not None:
            Ξ = (1 - self.r) * Ξ + self.r * self.AM_freq @ Ξ @ self.AM_mut
        return onp.float64(loss(Ξ, self.X))
