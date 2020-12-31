#! /usr/bin/env python

import mushi.histories as hst
from mushi import utils
import mushi.optimization as opt
import mushi.composition as cmp

from jax.config import config
import numpy as onp
import jax.numpy as np
from jax import jit, grad
from jax.ops import index, index_update
from jax.scipy.special import expit, logit
from scipy.stats import poisson
import prox_tv as ptv
from typing import Union, List, Dict
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
            if len(X.shape) == 1:
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
            raise ValueError('either file, or X, or n must be specified')
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

    def tmrca_cdf(self, eta: hst.eta) -> onp.ndarray:
        r"""The CDF of the TMRCA at each change point in ``eta``

        Args:
            eta: demographic history
        """
        if eta is None:
            raise ValueError('η(t) must be defined first')
        t, y = eta.arrays()
        return 1 - utils.tmrca_sf(t, y, self.n)[1:-1]

    def simulate(self, eta: hst.eta, mu: Union[hst.mu, np.float64],
                 r: np.float64 = 0,
                 seed: int = None) -> None:
        r"""Simulate a :math:`k`-SFS under the Poisson random field model
        (no linkage), assign to ``X`` attribute

        Args:
            eta: demographic history
            mu: mutation spectrum history (or constant rate)
            r: ancestral state misidentification rate (default 0)
            seed: random seed
        """
        onp.random.seed(seed)
        t, y = eta.arrays()
        M = utils.M(self.n, t, y)
        L = self.C @ M
        if type(mu) == hst.mu:
            if not eta.check_grid(mu):
                raise ValueError('η(t) and μ(t) must use the same time grid')
        else:
            mu = hst.mu(eta.change_points, mu * np.ones_like(y))
        Ξ = L @ mu.Z
        self.mutation_types = mu.mutation_types
        if len(self.mutation_types) == 1:
            self.AM_mut = np.array([[1]])
        else:
            self.AM_mut = utils.mutype_misid(self.mutation_types)
        self.X = poisson.rvs((1 - r) * Ξ + r * self.AM_freq @ Ξ @ self.AM_mut)

    def infer_history(self,  # noqa: C901
                      change_points: np.array,
                      mu0: np.float64,
                      eta: hst.eta = None,
                      eta_ref: hst.eta = None,
                      mu_ref: hst.mu = None,
                      infer_eta: bool = True,
                      infer_mu: bool = True,
                      alpha_tv: np.float64 = 0,
                      alpha_spline: np.float64 = 0,
                      alpha_ridge: np.float64 = 0,
                      beta_tv: np.float64 = 0,
                      beta_spline: np.float64 = 0,
                      beta_rank: np.float64 = 0,
                      hard: bool = False,
                      beta_ridge: np.float64 = 0,
                      max_iter: int = 1000,
                      s0: int = 1,
                      max_line_iter: int = 100,
                      gamma: np.float64 = 0.8,
                      tol: np.float64 = 0,
                      loss: str = 'prf',
                      verbose: bool = False,
                      folded: bool = False
                      ) -> None:
        r"""Perform sequential inference to fit :math:`\eta(t)` and
        :math:`\mu(t)`

        Args:
            change_points: epoch change points (ordered times > 0)
            mu0: total mutation rate (per genome per generation)
            eta: initial demographic history. By default, a
                 constant MLE is computed
            eta_ref: reference demographic history for ridge penalty. If
                     ``None``, the constant MLE is used
            mu_ref: reference MuSH for ridge penalty. If None, the constant
                    MLE is used
            infer_eta: perform :math:`\eta` inference if ``True``
            infer_mu: perform :math:`\mu` inference if ``True``
            loss: loss function, 'prf' for Poisson random field, 'kl' for
                  Kullback-Leibler divergence, 'lsq' for least-squares
            alpha_tv: total variation penalty on :math:`\eta(t)`
            alpha_spline: L2 on first differences penalty on :math:`\eta(t)`
            alpha_ridge: L2 for strong convexity penalty on :math:`\eta(t)`
            hard: hard rank penalty on :math:`\mu(t)` (non-convex)
            beta_tv: total variation penalty on :math:`\mu(t)`
            beta_spline: penalty on :math:`\mu(t)`
            beta_rank: rank penalty on :math:`\mu(t)`
            beta_ridge: L2 penalty on :math:`\mu(t)`
            max_iter: maximum number of proximal gradient steps
            tol: relative tolerance in objective function (if ``0``, not used)
            s0: max step size
            max_line_iter: maximum number of line search steps
            gamma: step size shrinkage rate for line search
            verbose: print verbose messages if ``True``
            folded: if ``False``, infer :math:`\eta(t)` using unfolded SFS. If
                    ``True``, can only be used with ``infer_mu=False``, and infer
                    :math:`\eta(t)` using folded SFS.
        """
        if folded is True and infer_mu is not False:
            raise ValueError('infer_mu=False is required for folded=True')
        assert self.X is not None, 'use simulate() to generate data first'
        if self.X is None:
            raise ValueError('use simulate() to generate data first')

        # pithify regularization parameter names
        α_tv = alpha_tv
        α_spline = alpha_spline
        α_ridge = alpha_ridge

        β_tv = beta_tv
        β_spline = beta_spline
        β_rank = beta_rank
        β_ridge = beta_ridge

        # ininitialize with MLE constant η and μ
        x = self.X.sum(1)

        # fold the spectrum if inference is on folded SFS
        if folded:
            x = utils.fold(x)

        μ_total = hst.mu(change_points,
                         mu0 * np.ones((len(change_points) + 1, 1)))
        t, z = μ_total.arrays()
        # number of segregating variants in each mutation type
        S = self.X.sum(0, keepdims=True)

        if eta is not None:
            self.η = eta
        elif self.η is None:
            # Harmonic number
            H = (1 / np.arange(1, self.n - 1)).sum()
            # constant MLE
            y = (S.sum() / 2 / H / mu0) * np.ones(len(z))
            self.η = hst.eta(change_points, y)

        μ_const = hst.mu(self.η.change_points,
                         mu0 * (S / S.sum()) * np.ones((self.η.m,
                                                        self.X.shape[1])),
                         mutation_types=self.mutation_types.values)

        if self.μ is None:
            self.μ = μ_const
        self.M = utils.M(self.n, t, self.η.y)
        self.L = self.C @ self.M

        # badness of fit
        if loss == 'prf':
            loss = utils.prf
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

        if infer_eta:
            if verbose:
                print('inferring η(t)', flush=True)

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
            logy_ref = np.log(eta_ref.y)

            # In the following, the parameter vector params will contain the
            # misid rate in params[0], and logy in params[1:]
            @jit
            def g(params):
                """differentiable piece of objective in η problem"""
                logy = params[1:]
                L = self.C @ utils.M(self.n, t, np.exp(logy))
                ξ = L @ z
                if folded:
                    ξ = utils.fold(ξ)
                else:
                    r = expit(params[0])
                    ξ = (1 - r) * ξ + r * self.AM_freq @ ξ
                loss_term = loss(np.squeeze(ξ), x)
                spline_term = (α_spline / 2) * ((D1 @ logy) ** 2).sum()
                # generalized Tikhonov
                logy_delta = logy - logy_ref
                ridge_term = (α_ridge / 2) * (logy_delta.T @ Γ @ logy_delta)
                return loss_term + spline_term + ridge_term

            @jit
            def h(params):
                """nondifferentiable piece of objective in η problem"""
                logy = params[1:]
                return α_tv * np.abs(D1 @ logy).sum()

            def prox(params, s):
                """total variation prox operator (no jit due to ptv module)"""
                if α_tv > 0:
                    params = index_update(params, index[1:],
                                          ptv.tv1_1d(params[1:], s * α_tv))
                return params

            # initial iterate
            params = np.concatenate((np.array([logit(1e-3)]), np.log(self.η.y)))

            params = opt.acc_prox_grad_method(params, g, jit(grad(g)), h,
                                              prox,
                                              tol=tol,
                                              max_iter=max_iter,
                                              s0=s0,
                                              max_line_iter=max_line_iter,
                                              gamma=gamma,
                                              verbose=verbose)
            if not folded:
                self.r = expit(params[0])
            y = np.exp(params[1:])
            self.η = hst.eta(self.η.change_points, y)
            self.M = utils.M(self.n, t, y)
            self.L = self.C @ self.M

        if infer_mu and len(self.mutation_types) > 1:
            if verbose:
                print('inferring μ(t) conditioned on η(t)', flush=True)

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
                Ξ = self.L @ (mu0 * cmp.ilr_inv(Z, basis))
                if self.r is not None:
                    Ξ = (1 - self.r) * Ξ + self.r * self.AM_freq @ Ξ @ self.AM_mut
                loss_term = loss(Ξ, self.X)
                spline_term = (β_spline / 2) * ((D1 @ Z) ** 2).sum()
                # generalized Tikhonov
                Z_delta = Z - Z_ref
                ridge_term = (β_ridge / 2) * np.trace(Z_delta.T @ Γ @ Z_delta)
                return loss_term + spline_term + ridge_term

            if β_tv and β_rank:
                @jit
                def h1(Z):
                    """1st nondifferentiable piece of objective in μ problem"""
                    return β_tv * np.abs(D1 @ Z).sum()

                shape = Z.T.shape
                w = β_tv * onp.ones(shape)
                w[:, -1] = 0
                w = w.flatten()[:-1]

                def prox1(Z, s):
                    """total variation prox operator on row dimension
                    """
                    return ptv.tv1w_1d(Z.T, s * w).reshape(shape).T

                @jit
                def h2(Z):
                    """2nd nondifferentiable piece of objective in μ problem"""
                    σ = np.linalg.svd(Z - Z_const, compute_uv=False)
                    return β_rank * np.linalg.norm(σ, 0 if hard else 1)

                def prox2(Z, s):
                    """singular value thresholding"""
                    U, σ, Vt = np.linalg.svd(Z - Z_const, full_matrices=False)
                    if hard:
                        σ = index_update(σ, index[σ <= s * β_rank], 0)
                    else:
                        σ = np.maximum(0, σ - s * β_rank)
                    Σ = np.diag(σ)
                    return Z_const + U @ Σ @ Vt

                Z = opt.three_op_prox_grad_method(Z, g, jit(grad(g)),
                                                  h1, prox1,
                                                  h2, prox2,
                                                  tol=tol,
                                                  max_iter=max_iter,
                                                  s0=s0,
                                                  max_line_iter=max_line_iter,
                                                  gamma=gamma, ls_tol=0,
                                                  verbose=verbose)

            else:
                if β_tv:
                    @jit
                    def h(Z):
                        """nondifferentiable piece of objective in μ problem"""
                        return β_tv * np.abs(D1 @ Z).sum()

                    shape = Z.T.shape
                    w = β_tv * onp.ones(shape)
                    w[:, -1] = 0
                    w = w.flatten()[:-1]

                    def prox(Z, s):
                        """total variation prox operator on row dimension
                        """
                        return ptv.tv1w_1d(Z.T, s * w).reshape(shape).T

                elif β_rank:
                    @jit
                    def h(Z):
                        """nondifferentiable piece of objective in μ problem"""
                        σ = np.linalg.svd(Z - Z_const, compute_uv=False)
                        return β_rank * np.linalg.norm(σ, 0 if hard else 1)

                    def prox(Z, s):
                        """singular value thresholding"""
                        U, σ, Vt = np.linalg.svd(Z - Z_const,
                                                 full_matrices=False)
                        if hard:
                            σ = index_update(σ, index[σ <= s * β_rank], 0)
                        else:
                            σ = np.maximum(0, σ - s * β_rank)
                        Σ = np.diag(σ)
                        return Z_const + U @ Σ @ Vt
                else:
                    @jit
                    def h(Z):
                        return 0

                    @jit
                    def prox(Z, s):
                        return Z

                Z = opt.acc_prox_grad_method(Z, g, jit(grad(g)), h,
                                             prox,
                                             tol=tol,
                                             max_iter=max_iter,
                                             s0=s0,
                                             max_line_iter=max_line_iter,
                                             gamma=gamma,
                                             verbose=verbose)

            self.μ = hst.mu(self.η.change_points,
                            mu0 * cmp.ilr_inv(Z, basis),
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
                z = self.μ.Z.sum(1)
            else:
                z = np.ones_like(self.η.y)
            ξ = self.L.dot(z)
            if folded:
                ξ = utils.fold(onp.array(ξ))
            else:
                if self.r is None:
                    raise ValueError('ancestral state misidentification rate '
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
