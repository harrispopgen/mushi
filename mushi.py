#! /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import numpy as np
from scipy.special import binom
from scipy.stats import poisson
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
import prox_tv as ptv


@dataclass
class History():
    '''base class piecewise constant history. The first epoch starts at zero,
    and the last epoch extends to infinity

    change_points: epoch change points (times)
    vals: constant values for each epoch (rows)
    '''
    change_points: np.array
    vals: np.ndarray

    def __post_init__(self):
        if any(np.diff(self.change_points) <= 0) or any(
           np.isinf(self.change_points)) or any(self.change_points <= 0):
            raise ValueError('change_points must be increasing, finite, and '
                             'positive')
        if len(self.vals) != len(self.change_points) + 1:
            raise ValueError(f'len(change_points) = {len(self.change_points)}'
                             f' implies {len(self.change_points) + 1} epochs,'
                             f' but len(vals) = {len(self.vals)}')
        if np.any(self.vals <= 0) or np.sum(np.isinf(self.vals)):
            raise ValueError(f'elements of vals must be finite and '
                             'positive')
        self.m = len(self.vals)

    def arrays(self):
        '''return time grid and values in each cell'''
        t = np.concatenate((np.array([0]),
                            self.change_points,
                            np.array([np.inf])))
        return t, self.vals

    def epochs(self):
        '''generator yielding epochs as tuples: (start_time, end_time, value)
        '''
        for i in range(self.m):
            if i == 0:
                start_time = 0
            else:
                start_time = self.change_points[i - 1]
            if i == self.m - 1:
                end_time = np.inf
            else:
                end_time = self.change_points[i]
            value = self.vals[i]
            yield (start_time, end_time, value)

    def check_grid(self, other: int):
        '''test if time grid is the same as another instance'''
        if any(self.change_points != other.change_points):
            return False
        else:
            return True

    def plot(self, idxs=None, **kwargs) -> None:
        '''plot the history

        idxs: indices of value column(s) to plot (optional)
        kwargs: key word arguments passed to plt.step
        '''
        t = np.concatenate((np.array([0]), self.change_points))
        if idxs is not None:
            vals = self.vals[:, idxs]
        else:
            vals = self.vals
        plt.step(t, vals, where='post', **kwargs)
        plt.xlabel('$t$')
        if 'label' in kwargs:
            plt.legend()


class η(History):
    '''demographic history

    change_points: epoch change points (times)
    y: vector of constant population sizes in each epoch
    '''
    @property
    def y(self):
        '''read-only alias to vals attribute in base class'''
        return self.vals

    @y.setter
    def y(self, value):
        self.vals = value

    def __post_init__(self):
        super().__post_init__()
        assert len(self.y.shape) == 1, self.y.shape

    def plot(self, **kwargs) -> None:
        super().plot(**kwargs)
        plt.xlabel('$t$')
        plt.ylabel('$η(t)$')
        plt.xscale('symlog')
        plt.yscale('log')
        plt.tight_layout()


class μ(History):
    '''mutation spectrum history

    change_points: epoch change points (times)
    Z: matrix of constant values for each epoch (rows) in each mutation type
       (columns)
    '''
    @property
    def Z(self):
        '''read-only alias to vals attribute in base class'''
        return self.vals

    @Z.setter
    def Z(self, value):
        self.vals = value

    def __post_init__(self):
        super().__post_init__()
        assert len(self.Z.shape) == 2, self.Z.shape

    def plot(self, idxs=None, **kwargs) -> None:
        super().plot(idxs=idxs, **kwargs)
        plt.xlabel('$t$')
        plt.ylabel('$\\mu(t)$')
        plt.xscale('symlog')
        plt.tight_layout()

    def heatmap(self):
        t = np.concatenate((np.array([0]), self.change_points))
        y, x = np.meshgrid(t, range(1, self.Z.shape[1] + 2))
        c = plt.pcolormesh(x, y, self.Z.T)
        for line in range(1, self.Z.shape[1] + 2):
            plt.axvline(line, c='k', lw=0.5)
        plt.gca().invert_yaxis()
        plt.xlabel('mutation type')
        plt.yscale('symlog')
        plt.ylabel('$t$', rotation=0)
        cbar = plt.colorbar(c)
        cbar.set_label('$\\mu(t)$', rotation=0)
        plt.tight_layout()


class kSFS():
    '''The kSFS model described in the text'''

    def __init__(self, η: η, X: np.ndarray = None, n: int = None):
        '''Sample frequency spectrum

        η: demographic history
        X: observed k-SFS matrix (optional)
        n: number of haplotypes (optional)
        '''
        self.η = η
        if X is not None:
            self.X = X
            self.n = len(X) + 1
        elif not n:
            raise ValueError('either x or n must be specified')
        else:
            self.n = n
        self.L = kSFS.C(self.n) @ kSFS.M(self.n, self.η)

    @staticmethod
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

    @staticmethod
    def M(n: int, η: η) -> np.ndarray:
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

    def simulate(self, μ: μ, seed: int = None) -> None:
        '''simulate a SFS under the Poisson random field model (no linkage)
        assigns simulated SFS to self.X

        μ: mutation spectrum history
        seed: random seed (optional)
        '''
        if not self.η.check_grid(μ):
            raise ValueError('η and μ histories must use the same time grid')
        np.random.seed(seed)
        self.X = poisson.rvs(self.L @ μ.Z)

    def ℓ(self, Z: np.ndarray, grad: bool = False) -> np.float:
        '''Poisson random field log-likelihood of history

        Z: mutation spectrum history matrix (μ.Z)
        grad: flag to also return gradient wrt Z
        '''
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        Ξ = self.L @ Z
        ℓ = poisson.logpmf(self.X, Ξ).sum()
        if grad:
            dℓdZ = self.L.T @ (self.X / Ξ - 1)
            return np.array([ℓ, dℓdZ])
        else:
            return ℓ

    def d_kl(self, Z: np.ndarray, grad: bool = False) -> float:
        '''Kullback-Liebler divergence between normalized SFS and its
        expectation under history
        ignores constant term

        Z: mutation spectrum history matrix (μ.Z)
        grad: flag to also return gradient wrt Z
        '''
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        X_normalized = self.X / self.X.sum(axis=0)
        Ξ = self.L @ Z
        Ξ_normalized = Ξ / Ξ.sum(axis=1)
        d_kl = (-X_normalized * np.log(Ξ_normalized)).sum()
        if grad:
            grad_d_kl = -self.L.T @ ((X_normalized / Ξ) * (1 - Ξ_normalized))
            return np.array([d_kl, grad_d_kl])
        else:
            return d_kl

    def lsq(self, Z: np.ndarray, grad: bool = False) -> float:
        '''least-squares loss between SFS and its expectation under history

        Z: mutation spectrum history matrix (μ.Z)
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

    def constant_μ_MLE(self) -> μ:
        '''gives the MLE for a constant μ history'''
        if self.X is None:
            raise ValueError('use simulate() to generate data first')
        z0 = self.X.sum(axis=0) / np.sum(self.L)
        return μ(self.η.change_points,
                 z0[np.newaxis, :] * np.ones((self.η.m, 1)))

    def infer_μ(self, λ_tv: np.float64 = 0, α_tv: np.float64 = 0,
                λ_r: np.float64 = 0, α_r: np.float64 = 0,
                γ: np.float64 = 0.8, steps: int = 1000, tol: np.float64 = 1e-4,
                fit='prf', bins: np.ndarray = None) -> μ:
        '''return inferred μ history given the sfs and η history

        λ_tv: fused LASSO regularization strength
        α_tv: relative penalty on L1 vs L2 in fused LASSO
        λ_r: spectral (rank) regularization strength
        α_r: relative penalty on L1 vs L2 in spectral regularization
        γ: step size shrinkage rate for line search
        steps: number of proximal gradient descent steps
        tol: relative tolerance in objective function
        fit: fit function, 'prf' for Poisson random field, 'kl' for
             Kullback-Leibler divergence, 'lsq' for least-squares
        '''
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
        if fit == 'prf':
            def misfit_func(*args, **kwargs):
                return -self.ℓ(*args, **kwargs)
        elif fit == 'kl':
            misfit_func = self.d_kl
        elif fit == 'lsq':
            misfit_func = self.lsq
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
            def prox_update(Z, s):
                '''spectral regularization (nuclear norm penalty)
                '''
                U, σ, Vt = np.linalg.svd(Z, full_matrices=False)
                Σ = np.diag(np.maximum(0, σ - s * λ_r * α_r))
                return U @ Σ @ Vt
        else:
            def prox_update(Z, s):
                return Z

        # Accelerated proximal gradient ascent: our loss function decomposes as
        # f = g + h, where g is differentiable and h is not.
        # https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
        # some matrices we'll need for the first difference penalties
        D = (np.eye(self.η.m, k=0) - np.eye(self.η.m, k=-1))
        W = np.eye(self.η.m)
        W[0, 0] = 0
        D1 = W @ D
        D2 = D.T @ D1

        def g(Z, grad=False):
            '''differentiable piece of loss'''
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
                return g, grad_g
            return g

        def h(Z):
            '''nondifferentiable piece of loss'''
            return λ_tv * α_tv * np.abs(D1 @ Z).sum() \
                + λ_r * α_r * np.linalg.norm(Z, ord='nuc')

        def f(Z):
            '''loss'''
            return g(Z) + h(Z)

        # initialize using constant μ history MLE
        μ = self.constant_μ_MLE()
        Z = μ.Z
        # our auxiliary iterates for acceleration
        Q = μ.Z
        # initial loss
        f_old = f(Z)
        # initial step size
        s = 1
        for k in range(1, steps + 1):
            # g(Q) and ∇g(Q)
            g1, grad_g1 = g(Q, grad=True)
            # Armijo line search
            while True:
                if not np.all(np.isfinite(grad_g1)):
                    raise RuntimeError(f'invalid gradient: {grad_g1}')
                # G_s(Q) as in the notes linked above
                G = (1 / s) * (Q - prox_update(Q - s * grad_g1, s))
                # test g(Q - sG_s(Q))
                g2 = g(Q - s * G)
                if g2 <= g1 - s * (grad_g1 * G).sum() \
                        + (s / 2) * (G ** 2).sum():
                    break
                else:
                    s *= γ
            # accelerated gradient step
            Z_old = Z
            Z = prox_update(Q - s * grad_g1, s)
            Q = Z + (k / (k + 3)) * (Z - Z_old)
            # Z = np.clip(Z, 1e-6, np.inf)
            if not np.all(np.isfinite(Z)) and np.all(Z > 0):
                raise RuntimeError(f'invalid Z value: {Z}')
            # terminate if loss function is constant within tolerance
            f_new = f(Z)
            rel_change = np.abs((f_new - f_old) / f_old)
            if rel_change < tol:
                print(f'relative change in loss function {rel_change:.2g} '
                      f'is within tolerance {tol} after {k} steps')
                break
            else:
                f_old = f_new
            if k == steps:
                print(f'step size limit {steps} reached with relative '
                      f'change in loss function {rel_change:.2g}')
            f_old = f_new
        if bins is not None:
            # restore stashed unbinned variables
            self.X = X_true
            self.L = L_true
        μ.Z = Z
        return μ

    def plot(self, i: int = None, μ: μ = None, prf_quantiles=False):
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

    def heatmap(self, μ: μ = None, linthresh=1):
        '''heatmap with mixed linear-log scale color bar

        μ: inferred mutation spectrum history, z-scores are shown if not None
        linthresh: the range within which the plot is linear (when μ = None)
        '''
        Y, X = np.meshgrid(range(1, self.n + 1), range(1, self.X.shape[1] + 2))
        if μ is None:
            Z = self.X.T
            cbar_label = 'number of variants'
            c = plt.pcolormesh(X, Y, Z, norm=SymLogNorm(linthresh))
            plt.yscale('symlog')
        else:
            Ξ = self.L @ μ.Z
            Z = (self.X.T - Ξ.T) / np.sqrt(Ξ.T)
            cbar_label = 'z-score'
            cmap_range = np.abs(Z).max()
            c = plt.pcolormesh(X, Y, Z, vmin=-cmap_range, vmax=cmap_range,
                               cmap='seismic')
        for line in range(1, self.X.shape[1] + 2):
            plt.axvline(line, c='k', lw=0.5)
        plt.gca().invert_yaxis()
        plt.xlabel('mutation type')
        plt.ylabel('sample frequency')
        cbar = plt.colorbar(c)
        cbar.set_label(cbar_label)
        plt.tight_layout()
