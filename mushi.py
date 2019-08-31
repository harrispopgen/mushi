#! /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import jax.numpy as np
import numpy as onp  # original numpy
from jax import jit, grad
from jax.experimental.optimizers import adagrad
from scipy.special import binom
from functools import lru_cache
from scipy.stats import poisson
from matplotlib import pyplot as plt
from matplotlib import figure
from scipy.optimize import minimize_scalar
import prox_tv as ptv

from jax.config import config
config.update("jax_enable_x64", True)

@dataclass(frozen=True)
class PiecewiseConstantHistory():
    '''The first epoch starts at zero, and the last epoch extends to infinity.
    Can be used for η or μ

    change_points: epoch change points (times)
    vals: vector of constant values for each epoch
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
        # if any(self.vals <= 0) or np.sum(np.isinf(self.vals)):
        #     raise ValueError('elements of vals must be finite and positive')

    def m(self):
        '''number of epochs
        '''
        return len(self.vals)

    def __hash__(self):
        '''needed for hashability
        '''
        return hash((tuple(self.change_points), tuple(self.vals)))

    def __eq__(self, other) -> bool:
        '''needed for hashability
        '''
        if any(self.change_points != other.change_points):
            return False
        if any(self.vals != other.vals):
            return False
        return True

    def plot(self, **kwargs) -> None:
        '''plot the history

        kwargs: key word arguments passed to plt.step
        '''
        t = np.concatenate((np.array([0]), self.change_points))
        plt.step(t, self.vals, where='post', **kwargs)
        if 'label' in kwargs:
            plt.legend()

    def arrays(self):
        t = np.concatenate((np.array([0]),
                            self.change_points,
                            np.array([np.inf])))
        return t, self.vals

@dataclass()
class JointHistory():
    '''Piecewise constant history of population size η and mutation rate μ.
    both histories must use the same time grid

    η: effective population size history
    μ: mutation rate history
    '''
    η: PiecewiseConstantHistory
    μ: PiecewiseConstantHistory

    def __post_init__(self):
        if any(self.η.change_points != self.μ.change_points):
            raise ValueError('η and μ histories must use the same time grid')

    def __hash__(self):
        '''needed for hashability
        '''
        return hash((tuple(self.η.change_points), tuple(self.η.vals),
                     tuple(self.μ.change_points), tuple(self.μ.vals)))

    def __eq__(self, other) -> bool:
        '''needed for hashability
        '''
        return self.η == other.η and self.μ == other.μ

    def plot(self, fig: figure.Figure = None, **kwargs) -> figure.Figure:
        '''plot the history

        fig: add to another figure
        kwargs: keyword arguments passed to plotting calls
        '''
        if fig:
            axes = fig.axes
        else:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
        plt.sca(axes[0])
        self.η.plot(**kwargs)
        plt.ylabel('$η(t)$')

        plt.sca(axes[1])
        self.μ.plot(**kwargs)
        plt.xlabel('$t$')
        plt.ylabel('$μ(t)$')

        return fig

    def arrays(self):
        t = np.concatenate((np.array([0]),
                            self.μ.change_points,
                            np.array([np.inf])))
        y = self.η.vals
        z = self.μ.vals
        return t, y, z


class SFS():
    '''The SFS model described in the text
    '''

    def __init__(self, n: int = None, x: np.ndarray = None):
        '''pass one of these arguments

        n: number of sampled haplotypes
        sfs: observed sfs vector
        '''
        if x is not None:
            self.x = x
            self.n = len(x) + 1
            assert n is None, 'pass only one of n or x'
        elif n is not None:
            self.x = None
            self.n = n
            assert x is None, 'pass only one of n or x'
        else:
            raise ValueError('must pass either n or x')
        if self.n < 2:
            raise ValueError('n must be larger than 1')
        self.C = SFS.C(self.n)
        self.binom_vec = binom(np.arange(2, self.n + 1), 2)
        self.binom_array_recip = np.diag(1 / self.binom_vec)

    @staticmethod
    def W1(n: int) -> np.ndarray:
        '''The W1 matrix defined in the text

        n: number of sampled haplotypes
        '''
        W1 = onp.zeros((n - 1, n - 1))
        b = onp.arange(1, n - 1 + 1)
        # j = 2
        W1[:, 0] = 6 / (n + 1)
        # j = 3
        W1[:, 1] = 10 * (5 * n - 6 * b - 4) / (n + 2) / (n + 1)
        for j in range(2, n - 1):
            col = j - 2
            # procedurally generated by Zeilberger's algorithm in Mathematica
            W1[:, col + 2] = -((-((-1 + j)*(1 + j)**2*(3 + 2*j)*(j - n)*(4 + 2*j - 2*b*j + j**2 - b*j**2 + 4*n + 2*j*n + j**2*n)*W1[:, col]) - (-1 + 2*j)*(3 + 2*j)*(-4*j - 12*b*j - 4*b**2*j - 6*j**2 - 12*b*j**2 - 2*b**2*j**2 - 4*j**3 + 4*b**2*j**3 - 2*j**4 + 2*b**2*j**4 + 4*n + 2*j*n - 6*b*j*n + j**2*n - 9*b*j**2*n - 2*j**3*n - 6*b*j**3*n - j**4*n - 3*b*j**4*n + 4*n**2 + 6*j*n**2 + 7*j**2*n**2 + 2*j**3*n**2 + j**4*n**2)*W1[:, col + 1])/(j**2*(2 + j)*(-1 + 2*j)*(1 + j + n)*(3 + b + j**2 - b*j**2 + 3*n + j**2*n)))
        return np.array(W1)

    @staticmethod
    def W2(n: int) -> np.ndarray:
        '''The W2 matrix defined in the text

        n: number of sampled haplotypes
        '''
        W2 = onp.zeros((n - 1, n - 1))
        b = onp.arange(1, n - 1 + 1)
        # j = 2
        W2[:, 0] = 0
        # j = 3
        W2[:, 1] = (20 * (n - 2)) / (n+1) / (n+2)
        for j in range(2, n - 1):
            col = j - 2
            # procedurally generated by Zeilberger's algorithm in Mathematica
            W2[:, col + 2] = ((-1 + j)*(1 + j)*(2 + j)*(3 + 2*j)*(j - n)*(1 + j - n)*(1 + j + n)*W2[:, col] + (-1 + 2*j)*(3 + 2*j)*(1 + j - n)*(j + n)*(2 - j - 2*b*j - j**2 - 2*b*j**2 + 2*n + j*n + j**2*n)*W2[:, col + 1])/((-1 + j)*j*(2 + j)*(-1 + 2*j)*(j - n)*(j + n)*(1 + j + n))
        return np.array(W2)

    @staticmethod
    def C(n: int) -> np.ndarray:
        '''The C matrix defined in the text

        n: number of sampled haplotypes
        '''
        return SFS.W1(n) - SFS.W2(n)

    @staticmethod
    def cumprod(X: np.ndarray) -> np.ndarray:
        '''Alas, jax doesn't differentiate np.cumprod

        X: 2D array
        '''
        return np.exp(np.cumsum(np.log(X)))

    @lru_cache(maxsize=1)
    def M(self, η) -> np.ndarray:
        '''The M matrix defined in the text

        η: η history
        '''
        t, y = η.arrays()
        # epoch durations
        s = np.diff(t)
        u = np.exp(-s / y)
        u = np.concatenate((np.array([1]), u))

        return self.binom_array_recip \
               @ SFS.cumprod(u[np.newaxis, :-1],
                         ) ** self.binom_vec[:, np.newaxis] \
               @ (np.eye(len(y), k=0) - np.eye(len(y), k=-1)) \
               @ np.diag(y)

    def tmrca_cdf(self, η: PiecewiseConstantHistory) -> np.ndarray:
        '''The cdf of the TMRCA of the sample at each change point

        η: η history
        '''
        t, y = η.arrays()
        # epoch durations
        s = np.diff(t)
        u = np.exp(-s / y)
        u = np.concatenate((np.array([1]), u))

        # the A_2j are the product of this matrix
        # NOTE: using letter  "l" as a variable name to match text
        l = onp.arange(2, self.n + 1)[:, np.newaxis]
        with onp.errstate(divide='ignore'):
            A2_terms = l * (l-1) / (l * (l-1) - l.T * (l.T-1))
        onp.fill_diagonal(A2_terms, 1)
        A2 = np.prod(A2_terms, axis=0)

        return 1 - (A2[np.newaxis, :] @ SFS.cumprod(u[np.newaxis, 1:-1]) ** self.binom_vec[:, np.newaxis]).T

    def ξ(self, history, use_cache=True) -> np.ndarray:
        '''expected sfs vector

        history: η and μ joint history
        '''
        z = history.μ.vals
        if use_cache:
            M = self.M(history.η)
        else:
            M = SFS.M.__wrapped__(self, history.η)
        return self.C @ M @ z

    def simulate(self, history: JointHistory, seed: int = None) -> None:
        '''simulate a SFS under the Poisson random field model (no linkage)

        history: η and μ joint history
        '''
        onp.random.seed(seed)
        self.x = poisson.rvs(self.ξ(history))

    def ℓ(self, history, use_cache=True) -> np.float64:
        '''Poisson random field log-likelihood of history, ignoring constant
        term

        history: η and μ joint history
        '''
        if self.x is None:
            raise ValueError('use simulate() to generate data first')
        ξ = self.ξ(history, use_cache=use_cache)
        ℓ = np.sum(self.x * np.log(ξ) - ξ)
        return ℓ

    def constant_η_MLE(self, μ: PiecewiseConstantHistory
                       ) -> PiecewiseConstantHistory:
        '''gives the MLE for a constant η history

        μ: μ history
        '''
        @jit
        def f(y0):
            y = y0 * np.ones_like(μ.vals)
            η = PiecewiseConstantHistory(μ.change_points, y)
            history = JointHistory(η, μ)
            ξ = self.ξ(history, use_cache=False)
            return -np.sum(self.x * np.log(ξ) - ξ)
        y0 = minimize_scalar(f, bounds=(0, None)).x
        return PiecewiseConstantHistory(μ.change_points, y0 * np.ones(μ.m()))

    def constant_μ_MLE(self, η: PiecewiseConstantHistory
                       ) -> PiecewiseConstantHistory:
        '''gives the MLE for a constant μ history

        η: η history
        '''
        if self.x is None:
            raise ValueError('use simulate() to generate data first')
        z0 = (self.x.sum() / np.sum(SFS.C(self.n) @ self.M(η)))
        return PiecewiseConstantHistory(η.change_points, z0 * np.ones(η.m()))

    def infer_η(self, μ: PiecewiseConstantHistory, λ: np.float64 = 0,
                α: np.float64 = .99, s: np.float64 = .01,
                steps: int = 100) -> PiecewiseConstantHistory:
        '''infer the η history given the sfs and μ history

        μ: μ history
        λ: regularization strength
        α: relative penalty on L1 vs L2
        s: step size parameter for proximal gradient descent
        steps: number of proximal gradient descent steps
        '''
        assert λ >= 0, 'λ must be nonnegative'
        assert 0 <= α <= 1, 'α must be in the unit interval'

        D = (np.eye(μ.m(), k=0) - np.eye(μ.m(), k=-1))

        # initialize using constant η history MLE
        η0 = self.constant_η_MLE(μ)
        y = η0.vals

        # differentiable piece of loss function
        @jit
        def f(y):
            η = PiecewiseConstantHistory(μ.change_points, y)
            history = JointHistory(η, μ)
            return -self.ℓ(history, use_cache=False) \
                   + (1 / 2) * λ * (1 - α) * np.sum((D @ y) ** 2)
        grad_f = jit(grad(f))
        for _ in range(steps):
            g = grad_f(y)
            if not all(np.isfinite(g)):
                raise RuntimeError(f'invalid gradient: {g}')
            y = y - s * g
            if α > 0:
                y = ptv.tv1_1d(y, λ * α)
            y = np.clip(y, 0., np.inf)
            if not all(np.isfinite(y)):
                raise RuntimeError(f'invalid y value: {y}')
        return PiecewiseConstantHistory(μ.change_points, y)

    def infer_μ(self, η: PiecewiseConstantHistory, λ: np.float64 = 0,
                α: np.float64 = .99, s: np.float64 = .01,
                steps: int = 100) -> PiecewiseConstantHistory:
        '''infer the μ history given the sfs and η history

        η: η history
        λ: regularization strength
        α: relative penalty on L1 vs L2
        s: step size parameter for proximal gradient descent
        steps: number of proximal gradient descent steps
        '''
        assert λ >= 0, 'λ must be nonnegative'
        assert 0 <= α <= 1, 'α must be in the unit interval'

        D = (np.eye(η.m(), k=0) - np.eye(η.m(), k=-1))

        # differentiable piece of loss function
        @jit
        def f(z):
            history = JointHistory(η,
                                   PiecewiseConstantHistory(η.change_points, z)
                                   )
            return -self.ℓ(history) \
                + (1 / 2) * λ * (1 - α) * np.sum((D @ z) ** 2)
        grad_f = jit(grad(f))
        # initialize using constant μ history MLE
        z = self.constant_μ_MLE(η).vals
        for _ in range(steps):
            g = grad_f(z)
            if not all(np.isfinite(g)):
                raise RuntimeError(f'invalid gradient: {g}')
            z = z - s * g
            if α > 0:
                z = ptv.tv1_1d(z, λ * α)
            z = np.clip(z, 0., np.inf)
            if not all(np.isfinite(z)):
                raise RuntimeError(f'invalid z value: {z}')


        return PiecewiseConstantHistory(η.change_points, z)

    def plot(self, history: JointHistory = None):
        '''plot the SFS data and optionally the expected SFS under history

        history: joint η and μ history
        '''
        if history is not None:
            ξ = self.ξ(history)
            ξ_lower = poisson.ppf(.025, ξ)
            ξ_upper = poisson.ppf(.975, ξ)
            plt.plot(range(1, self.n), ξ, 'r--', label=r'$\xi$')
            plt.fill_between(range(1, self.n),
                             ξ_lower, ξ_upper,
                             facecolor='r', alpha=0.25,
                             label='inner 95%\nquantile')
        plt.plot(range(1, len(self.x) + 1), self.x,
                 'k.', alpha=.25, label=r'data')
        plt.xlabel('$b$')
        plt.ylabel(r'$ξ_b$')
        plt.xscale('log')
        plt.yscale('symlog')
        plt.legend()
