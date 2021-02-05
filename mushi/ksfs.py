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
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

config.update('jax_enable_x64', True)
# config.update('jax_debug_nans', True)


class kSFS():
    r"""Primary class for working with SFS data to infer demography
    :math:`\eta(t)\equiv 2N(t)`, or with :math:`k`-SFS data to infer demography
    and mutation spectrum history :math:`\boldsymbol\mu(t)`.

    Attributes:
        X (:obj:`numpy.ndarray`): :math:`k`-SFS matrix (or 1D SFS vector)
        eta (:obj:`mushi.eta`): demographic history
        mu (:obj:`mushi.mu`): mutation spectrum history
        mutation_types (:obj:`List[str]`): mutation spectrum history
        n (:obj:`int`): number of sampled haplotypes

    Args:
        ksfs_file: path to :math:`k`-SFS file, as ouput by ``mutyper ksfs``
        X: :math:`k`-SFS matrix
        mutation_types: list of names of X columns
        n: number of sampled haplotypes

    Examples:

        >>> import mushi
        >>> import numpy as np

        Three constructors:

        1. ``ksfs_file``: path to k-SFS file, as ouput by `mutyper ksfs`

        >>> ksfs = mushi.kSFS(file='ksfs.tsv') # doctest: +SKIP

        2. ``X`` and ``mutation_types`` (the latter may be ommitted if ``X`` is
           a 1D SFS array)

        >>> sfs = mushi.kSFS(X=np.array([10, 5, 3, 1]))

        >>> ksfs = mushi.kSFS(X=np.ones((10, 4)),
        ...                   mutation_types=['AAA>ACA', 'ACA>AAA',
        ...                                   'TCC>TTC', 'GAA>GGA'])

        3. ``n``: number of haplotypes to initialize for simulation

        >>> ksfs = mushi.kSFS(n=100)
    """

    def __init__(self, X: np.ndarray = None, mutation_types: List[str] = None,
                 file: str = None,
                 n: int = None):
        if file is not None:
            df = pd.read_csv(file, sep='\t', index_col=0)
            assert np.all(df.values >= 0)
            n = df.shape[0] + 1
            self.X = np.array(df.values)  # NOTE: np is jax.numpy
            self.n = len(self.X) + 1
            self.mutation_types = pd.Index(df.columns,
                                           name='mutation type')

        elif X is not None:
            self.X = np.array(X)  # NOTE: np is jax.numpy
            self.n = len(X) + 1
            if self.X.ndim == 2:
                if mutation_types is None:
                    raise TypeError('must specify mutation_types')
                if len(mutation_types) != self.X.shape[1]:
                    raise ValueError('inconsistent number of mutation '
                                     f'types {len(mutation_types)} for X '
                                     f'with {self.X.shape[1]} columns')
                self.mutation_types = pd.Index(mutation_types,
                                               name='mutation type')
        elif n is None:
            raise TypeError('either file, or X, or n must be specified')
        else:
            self.n = n
            self.X = None
            self.mutation_types = None
        self.C = utils.C(self.n)
        self.η = None
        self.μ = None
        self.M = None
        self.L = None

        # ancestral state misidentification
        # misidentification rate
        self.r = None
        # mutation type-wise misidentification rate
        self.r_vector = None
        # frequency misidentification operator
        self.AM_freq = np.eye(self.n - 1)[::-1]
        # mutation type misidentification operator
        if self.X is not None and self.X.ndim == 2:
            self.AM_mut = utils.mutype_misid(self.mutation_types)

    @property
    def eta(self) -> hst.eta:
        r"""Read-only alias to η attribute
        """
        return self.η

    def _check_eta(self):
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
        index = pd.Index(range(1, self.n), name='sample frequency')
        if self.X.ndim == 1:
            return pd.Series(self.X, index=index, name='SFS')
        elif self.X.ndim == 2:
            return pd.DataFrame(self.X, index=index,
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
            self._check_eta()
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

        Examples:

           Define sample size:

           >>> ksfs = mushi.kSFS(n=10)

           Define demographic history and MuSH:

           >>> eta = mushi.eta(np.array([1, 100, 10000]), np.array([1e4, 1e4, 1e2, 1e4]))
           >>> mush = mushi.mu(eta.change_points, np.ones((4, 4)),
           ...                 ['AAA>ACA', 'ACA>AAA', 'TCC>TTC', 'GAA>GGA'])

           Define ancestral misidentification rate:

           >>> r = 0.03

           Set random seed:

           >>> seed = 0

           Run simulation and print simulated :math:`k`-SFS

           >>> ksfs.simulate(eta, mush, r, seed)

           >>> ksfs.as_df() # doctest: +NORMALIZE_WHITESPACE
           mutation type     AAA>ACA  ACA>AAA  TCC>TTC  GAA>GGA
           sample frequency
           1                    1118     1123     1106     1108
           2                     147      128      120       98
           3                      65       55       66       60
           4                      49       52       64       46
           5                      44       43       34       36
           6                      35       28       36       33
           7                      23       32       24       35
           8                      34       32       24       24
           9                      52       41       57       56
        """
        onp.random.seed(seed)
        t, y = eta.arrays()
        M = utils.M(self.n, t, y)
        L = self.C @ M
        if type(mu) == hst.mu:
            eta.check_grid(mu)
            Ξ = L @ mu.Z
            self.mutation_types = mu.mutation_types
            self.AM_mut = utils.mutype_misid(self.mutation_types)
            self.X = np.array(poisson.rvs((1 - r) * Ξ
                                          + r * self.AM_freq @ Ξ @ self.AM_mut)
                              )
        else:
            ξ = mu * L.sum(1)
            self.X = np.array(poisson.rvs((1 - r) * ξ + r * self.AM_freq @ ξ))

    def set_eta(self, eta: hst.eta):
        r"""Set pre-specified demographic history :math:`\eta(t)`

        Args:
            eta: demographic history object
        """
        self.η = eta
        t = self.η.arrays()[0]
        self.M = utils.M(self.n, t, self.η.y)
        self.L = self.C @ self.M
        self.r = 0

    def infer_misid(self,
                    mu0: np.float64,
                    loss: str = 'prf',
                    max_iter: int = 100,
                    tol: np.float64 = 0,
                    line_search_kwargs: Dict = {},
                    verbose: bool = False
                    ) -> None:
        r"""Infer ancestral misidentification rate with :math:`\eta(t)` fixed.
        This function is used for fitting with a pre-specified demography, after
        using (see :func:`~mushi.kSFS.set_eta`), instead of inferring
        :math:`\eta(t)` with :func:`~mushi.kSFS.infer_eta` (which jointly infers
        misidentification rate).

        Args:
            mu0: total mutation rate (per genome per generation)
            loss: loss function name from :mod:`~mushi.loss_functions` module
            max_iter: maximum number of optimization steps
            tol: relative tolerance in objective function (if ``0``, not used)
            line_search_kwargs: line search keyword arguments,
                                see :py:meth:`mushi.optimization.LineSearcher`
            verbose: print verbose messages if ``True``
        """
        self._check_eta()
        if self.X is None:
            raise TypeError('use simulate() to generate data first')

        # total SFS
        if self.X.ndim == 1:
            x = self.X
        else:
            x = self.X.sum(1)

        # badness of fit
        loss = getattr(loss_functions, loss)

        self.mu0 = mu0

        @jit
        def g(r_logit):
            """differentiable piece of objective in η problem"""
            ξ = self.mu0 * self.L.sum(1)
            r = expit(r_logit)
            ξ = (1 - r) * ξ + r * self.AM_freq @ ξ
            return loss(np.squeeze(ξ), x)

        @jit
        def h(params):
            return 0

        @jit
        def prox(params, s):
            return params

        # optimizer
        optimizer = opt.AccProxGrad(g, jit(grad(g)), h, prox,
                                    verbose=verbose, **line_search_kwargs)
        # initial point
        r_logit = np.array([logit(1e-2)])
        # run optimization
        r_logit = optimizer.run(r_logit, tol=tol, max_iter=max_iter)

        self.r = expit(r_logit)

    def infer_eta(self,
                  mu0: np.float64,
                  *trend_penalty: Tuple[int, np.float64],
                  ridge_penalty: np.float64 = 0,
                  folded: bool = False,
                  pts: np.float64 = 100,
                  ta: np.float64 = None,
                  log_transform: bool = True,
                  eta: hst.eta = None,
                  eta_ref: hst.eta = None,
                  loss: str = 'prf',
                  max_iter: int = 100,
                  tol: np.float64 = 0,
                  line_search_kwargs: Dict = {},
                  trend_kwargs: Dict = {},
                  verbose: bool = False
                  ) -> None:
        r"""Infer demographic history :math:`\eta(t)`

        Args:
            mu0: total mutation rate (per genome per generation)
            trend_penalty: tuple ``(k, λ)`` for :math:`k`-th order trend
                           penalty (can pass multiple for mixed trends)
            ridge_penalty: ridge penalty
            folded: if ``False``, infer :math:`\eta(t)` using unfolded SFS. If
                    ``True``, can only be used with ``infer_mu=False``, and
                    infer :math:`\eta(t)` using folded SFS.
            pts: number of points for time discretization
            ta: time (in WF generations ago) of oldest change point in time
                discretization. If ``None``, set automatically based on
                10 * E[TMRCA] under MLE constant demography
            log_transform: fit :math:`\log\eta(t)`
            eta: initial demographic history. By default, a constant MLE is
                 computed
            eta_ref: reference demographic history for ridge penalty. If
                     ``None``, the constant MLE is used
            loss: loss function name from :mod:`~mushi.loss_functions` module
            max_iter: maximum number of optimization steps
            tol: relative tolerance in objective function (if ``0``, not used)
            line_search_kwargs: line search keyword arguments,
                                see :py:meth:`mushi.optimization.LineSearcher`
            trend_kwargs: keyword arguments for trend filtering,
                          see :py:meth:`mushi.optimization.TrendFilter.run`
            verbose: print verbose messages if ``True``

        Examples:

            Suppose ``ksfs`` is a ``kSFS`` object. Then the following fits a
            demographic history with 0-th order (piecewise constant) trend
            penalization of strength 100, assuming a mutation rate of 10
            mutations per genome per generation.

            >>> mu0 = 1
            >>> ksfs.infer_eta(mu0, (0, 1e2))

            Alternatively, a mixed trend solution, with constant and cubic
            pieces, is fit with

            >>> ksfs.infer_eta(mu0, (0, 1e2), (3, 1e1))

            The attribute ``ksfs.eta`` is now set and accessable for plotting
            (see :class:`~mushi.eta`).
        """
        if self.X is None:
            raise TypeError('use simulate() to generate data first')

        # total SFS
        if self.X.ndim == 1:
            x = self.X
        else:
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
            self.set_eta(eta)
        elif self.η is None:
            y = N_const * np.ones(change_points.size + 1)
            self.η = hst.eta(change_points, y)
        t = self.η.arrays()[0]

        self.mu0 = mu0

        # badness of fit
        loss = getattr(loss_functions, loss)

        # Accelerated proximal gradient method: our objective function
        # decomposes as f = g + h, where g is differentiable and h is not.
        # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

        # rescale trend penalties to be comparable between orders and time grids
        # filter zeros from trend penalties
        trend_penalty = tuple((k, (self.η.m ** k / onp.math.factorial(k)) * λ)
                              for k, λ in trend_penalty if λ > 0)

        # Tikhonov matrix
        if eta_ref is None:
            eta_ref = self.η
            Γ = np.diag(np.ones_like(eta_ref.y))
        else:
            self.η.check_grid(eta_ref)
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
                       for k, λ in trend_penalty)

        def prox(params, s):
            """trend filtering prox operator (no jit due to ptv module)"""
            if trend_penalty:
                k, sλ = zip(*((k, s * λ) for k, λ in trend_penalty))
                trend_filterer = opt.TrendFilter(k, sλ)
                params = params.at[1:].set(trend_filterer.run(params[1:],
                                                              **trend_kwargs))
            if log_transform:
                return params
            # else:
            # clip to minimum population size of 1
            return np.clip(params, 1)

        # optimizer
        optimizer = opt.AccProxGrad(g, jit(grad(g)), h, prox,
                                    verbose=verbose, **line_search_kwargs)
        # initial point
        params = np.concatenate((np.array([logit(1e-2)]),
                                 np.log(self.η.y)
                                 if log_transform else self.η.y))
        # run optimization
        params = optimizer.run(params, tol=tol, max_iter=max_iter)

        if not folded:
            self.r = expit(params[0])
        y = np.exp(params[1:]) if log_transform else params[1:]
        self.η = hst.eta(self.η.change_points, y)
        self.M = utils.M(self.n, t, y)
        self.L = self.C @ self.M

    def infer_mush(self,
                   *trend_penalty: Tuple[int, np.float64],
                   ridge_penalty: np.float64 = 0,
                   rank_penalty: np.float64 = 0,
                   hard: bool = False,
                   mu_ref: hst.mu = None,
                   misid_penalty: np.float64 = 1e-4,
                   loss: str = 'prf',
                   max_iter: int = 100,
                   tol: np.float64 = 0,
                   line_search_kwargs: Dict = {},
                   trend_kwargs: Dict = {},
                   verbose: bool = False
                   ) -> None:
        r"""Infer mutation spectrum history :math:`\mu(t)`

        Args:
            trend_penalty: tuple ``(k, λ)`` for :math:`k`-th order trend
                           penalty (can pass multiple for mixed trends)
            ridge_penalty: ridge penalty
            rank_penalty: rank penalty
            hard: hard rank penalty (non-convex)
            mu_ref: reference MuSH for ridge penalty. If None, the constant
                    MLE is used
            misid_penalty: ridge parameter to shrink misid rates to aggregate
                           rate
            loss: loss function from :mod:`~mushi.loss_functions` module
            max_iter: maximum number of optimization steps
            tol: relative tolerance in objective function (if ``0``, not used)
            line_search_kwargs: line search keyword arguments,
                                see :py:class:`mushi.optimization.LineSearcher`
            trend_kwargs: keyword arguments for trend filtering,
                          see :py:meth:`mushi.optimization.TrendFilter.run`
            verbose: print verbose messages if ``True``

        Examples:
            Suppose ``ksfs`` is a ``kSFS`` object, and the demography has
            already been fit with ``infer_eta``. The following fits a
            mutation spectrum history with 0-th order (piecewise constant) trend
            penalization of strength 100.

            >>> ksfs.infer_mush((0, 1e2))

            Alternatively, a mixed trend solution, with constant and cubic
            pieces, and with rank penalization 100, is fit with

            >>> ksfs.infer_mush((0, 1e2), (3, 1e1), rank_penalty=1e2)

            The attribute ``ksfs.mu`` is now set and accessable for plotting
            (see :class:`~mushi.mu`).
        """
        if self.X is None:
            raise TypeError('use simulate() to generate data first')
        self._check_eta()
        if self.r is None or self.r == 0:
            raise ValueError('ancestral misidentification rate has not been '
                             'inferred, possibly due to folded SFS inference')
        if self.mutation_types is None:
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

        # rescale trend penalties to be comparable between orders and time grids
        # filter zeros from trend penalties
        trend_penalty = tuple((k, (self.μ.m ** k / onp.math.factorial(k)) * λ)
                              for k, λ in trend_penalty if λ > 0)

        if mu_ref is None:
            mu_ref = μ_const
            # Tikhonov matrix
            Γ = np.diag(np.ones_like(self.η.y))
        else:
            self.μ.check_grid(mu_ref)
            # - log(1 - CDF)
            Γ = np.diag(- np.log(utils.tmrca_sf(t, self.η.y, self.n))[:-1])

        # orthonormal basis for Aitchison simplex
        # NOTE: instead of Gram-Schmidt could try SVD of clr transformed X
        #       https://en.wikipedia.org/wiki/Compositional_data#Isometric_logratio_transform
        basis = cmp._gram_schmidt_basis(self.μ.Z.shape[1])
        check_orth = True if self.μ.Z.shape[1] > 2 else False

        # constand MLE and reference mush
        Z_const = cmp.ilr(μ_const.Z, basis, check_orth)
        Z_ref = cmp.ilr(mu_ref.Z, basis, check_orth)

        # weights for relating misid rates to aggregate misid rate from eta step
        misid_weights = self.X.sum(0) / self.X.sum()
        # reference composition for weighted misid (if all rates are equal)
        misid_ref = cmp.ilr(misid_weights, basis, check_orth)

        # In the following, params will hold the weighted misid composition in
        # the first row and the mush composition at each time in the remaining rows

        @jit
        def g(params):
            """differentiable piece of objective in μ problem"""
            r = self.r * cmp.ilr_inv(params[0, :], basis) / misid_weights
            Z = params[1:, :]
            Ξ = self.L @ (self.mu0 * cmp.ilr_inv(Z, basis))
            Ξ =  Ξ * (1 - r) + self.AM_freq @ Ξ @ (self.AM_mut * r[:, np.newaxis])
            loss_term = loss(Ξ, self.X)
            Z_delta = Z - Z_ref
            ridge_term = (ridge_penalty / 2) * np.sum(Z_delta * (Γ @ Z_delta))
            misid_delta = params[0, :] - misid_ref
            misid_ridge_term = misid_penalty * np.sum(misid_delta ** 2)
            return loss_term + ridge_term + misid_ridge_term

        if trend_penalty:

            @jit
            def h_trend(params):
                """trend filtering penalty"""
                return sum(λ * np.linalg.norm(np.diff(params[1:, :], k,
                                                      axis=0), 1)
                           for k, λ in trend_penalty)

            def prox_trend(params, s):
                """trend filtering prox operator (no jit due to ptv module)"""
                k, sλ = zip(*((k, s * λ) for k, λ in trend_penalty))
                trend_filterer = opt.TrendFilter(k, sλ)
                return params.at[1:, :].set(trend_filterer.run(params[1:, :],
                                              **trend_kwargs))

        if rank_penalty:
            if self.mutation_types.size < 3:
                raise ValueError('kSFS must have more than 2 mutation types for'
                                 ' rank penalization')

            @jit
            def h_rank(params):
                """2nd nondifferentiable piece of objective in μ problem"""
                if hard:
                    return rank_penalty * np.linalg.matrix_rank(params[1:, :]
                                                                - Z_const)
                # else:
                return rank_penalty * np.linalg.norm(params[1:, :] - Z_const,
                                                     'nuc')

            def prox_rank(params, s):
                """singular value thresholding"""
                U, σ, Vt = np.linalg.svd(params[1:, :] - Z_const,
                                         full_matrices=False)
                if hard:
                    σ = σ.at[σ <= s * rank_penalty].set(0)
                else:
                    σ = np.maximum(0, σ - s * rank_penalty)
                Σ = np.diag(σ)
                return params.at[1:, :].set(Z_const + U @ Σ @ Vt)

            if not hard:
                prox_rank = jit(prox_rank)

        # optimizer
        if trend_penalty and rank_penalty:
            optimizer = opt.ThreeOpProxGrad(g, jit(grad(g)),
                                            h_trend, prox_trend,
                                            h_rank, prox_rank,
                                            verbose=verbose,
                                            **line_search_kwargs)
        else:
            if trend_penalty:
                h = h_trend
                prox = prox_trend
            elif rank_penalty:
                h = h_rank
                prox = prox_rank
            else:
                @jit
                def h(params):
                    return 0

                @jit
                def prox(params, s):
                    return params

            optimizer = opt.AccProxGrad(g, jit(grad(g)), h, prox,
                                        verbose=verbose, **line_search_kwargs)

        # initial point (note initial row is for misid rates)
        # ---------------------------------------------------
        params = np.zeros((self.μ.m + 1, self.mutation_types.size - 1))
        # misid rate for each mutation type
        if self.r_vector is not None:
            r = self.r_vector
        else:
            r = self.r * np.ones(self.mutation_types.size)
        params = params.at[0, :].set(cmp.ilr(misid_weights * r, basis, check_orth))
        # ilr transformed mush
        ilr_mush = cmp.ilr(self.μ.Z, basis, check_orth)
        # make sure it's a column vector if only 2 mutation types
        if ilr_mush.ndim == 1:
            ilr_mush = ilr_mush[:, np.newaxis]
        params = params.at[1:, :].set(ilr_mush)
        # ---------------------------------------------------

        # run optimizer
        params = optimizer.run(params, tol=tol, max_iter=max_iter)

        # update attributes
        self.r_vector = self.r * cmp.ilr_inv(params[0, :], basis) / misid_weights
        self.μ = hst.mu(self.η.change_points,
                        self.mu0 * cmp.ilr_inv(params[1:, :], basis),
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
        if self.X.ndim == 1:
            x = self.X
        else:
            x = self.X.sum(1)
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
                ξ = utils.fold(ξ)
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
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
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
        if self.η is not None and self.r is None:
            print('warning: misidentification rate is not defined, perhaps due'
                  ' to folded SFS inference, and will be set to 0 for plotting')
            r = 0
        else:
            r = self.r
        if self.μ is not None:
            Ξ = self.L @ self.μ.Z
            Ξ = Ξ * (1 - self.r_vector) + self.AM_freq @ Ξ @ (self.AM_mut * self.r_vector[:, np.newaxis])
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
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
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
            func: loss function name from :mod:`~mushi.loss_functions` module

        Example:
            After fitting demography and/or MuSH (with ``infer_eta`` and
            ``infer_mush``), the loss (goodness of fit) may be evaluated as

        >>> ksfs.loss()
        -31584.277426010947
        """
        self._check_eta()
        loss = getattr(loss_functions, func)
        if self.μ is None:
            ξ = self.mu0 * self.L.sum(1)
            if self.r is not None:
                ξ = utils.fold(ξ)
                x = utils.fold(self.X.sum(1))
            else:
                ξ = (1 - self.r) * ξ + self.r * self.AM_freq @ ξ
            return loss(np.squeeze(ξ), np.squeeze(x))
        # else:
        Ξ = self.L @ self.μ.Z
        Ξ = Ξ * (1 - self.r_vector) + self.AM_freq @ Ξ @ (self.AM_mut * self.r_vector[:, np.newaxis])
        return onp.float64(loss(Ξ, self.X))
