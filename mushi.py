#! /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import numpy as np
from scipy.special import binom
from functools import lru_cache
from scipy.stats import poisson
from matplotlib import pyplot as plt
from matplotlib import figure
from scipy.optimize import minimize_scalar
import prox_tv as ptv


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

    def epochs(self):
        '''generator yielding epochs in history as tuples:
        (start_time, end_time, value)
        '''
        for i in range(self.m()):
            if i == 0:
                start_time = 0
            else:
                start_time = self.change_points[i - 1]
            if i == self.m() - 1:
                end_time = np.inf
            else:
                end_time = self.change_points[i]
            value = self.vals[i]
            yield (start_time, end_time, value)

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
        self._binom_vec = binom(np.arange(2, self.n + 1), 2)
        self._binom_array_recip = np.diag(1 / self._binom_vec)

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

        return self._binom_array_recip \
               @ np.cumprod(u[np.newaxis, :-1],
                            axis=1) ** self._binom_vec[:, np.newaxis] \
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
        l = np.arange(2, self.n + 1)[:, np.newaxis]
        with np.errstate(divide='ignore'):
            A2_terms = l * (l-1) / (l * (l-1) - l.T * (l.T-1))
        np.fill_diagonal(A2_terms, 1)
        A2 = np.prod(A2_terms, axis=0)

        return 1 - (A2[np.newaxis, :]
                    @ np.cumprod(u[np.newaxis, 1:-1],
                                 axis=1) ** self._binom_vec[:, np.newaxis]).T

    def ξ(self, history: JointHistory, jac: bool = False) -> np.ndarray:
        '''expected sfs vector

        history: η and μ joint history
        jac: flag to return jacobian wrt μ
        '''
        z = history.μ.vals
        L = self.C @ self.M(history.η)
        if jac:
            return L @ z, L
        return L @ z

    def simulate(self, history: JointHistory, seed: int = None) -> None:
        '''simulate a SFS under the Poisson random field model (no linkage)

        history: η and μ joint history
        '''
        np.random.seed(seed)
        self.x = poisson.rvs(self.ξ(history))

    def ℓ(self, history: JointHistory, grad: bool = False) -> np.float:
        '''Poisson random field log-likelihood of history

        history: η and μ joint history
        grad: flag to return gradient wrt μ
        '''
        if self.x is None:
            raise ValueError('use simulate() to generate data first')
        if grad:
            ξ, J_μξ = self.ξ(history, jac=True)
            dℓdμ = J_μξ.T @ (self.x / ξ - 1)
            return dℓdμ
        else:
            return poisson.logpmf(self.x, self.ξ(history)).sum()

    def d_kl(self, history: JointHistory, grad: bool = False) -> float:
        '''Kullback-Liebler divergence between normalized SFS and its
        expectation under history
        ignores constant term

        history: η and μ joint history
        grad: flag to return gradient wrt μ
        '''
        if self.x is None:
            raise ValueError('use simulate() to generate data first')
        x_normalized = self.x / self.x.sum()
        if grad:
            ξ, J_μξ = self.ξ(history, jac=True)
            ξ_normalized = ξ / ξ.sum()
            return -J_μξ.T @ ((x_normalized / ξ) * (1 - ξ_normalized))
        else:
            ξ = self.ξ(history)
            ξ_normalized = ξ / ξ.sum()
            return -x_normalized.dot(np.log(ξ_normalized))

    def lsq(self, history: JointHistory, grad: bool = False) -> float:
        '''least-squares loss between SFS and its expectation under history

        history: η and μ joint history
        grad: flag to return gradient wrt μ
        '''
        if self.x is None:
            raise ValueError('use simulate() to generate data first')
        if grad:
            ξ, J_μξ = self.ξ(history, jac=True)
            return J_μξ.T @ (ξ - self.x)
        else:
            return (1 / 2) * ((self.ξ(history) - self.x) ** 2).sum()


    def constant_μ_MLE(self, η: PiecewiseConstantHistory
                       ) -> PiecewiseConstantHistory:
        '''gives the MLE for a constant μ history

        η: η history
        '''
        if self.x is None:
            raise ValueError('use simulate() to generate data first')
        z0 = (self.x.sum() / np.sum(self.C @ self.M(η)))
        return PiecewiseConstantHistory(η.change_points, z0 * np.ones(η.m()))

    def infer_μ(self, η: PiecewiseConstantHistory, λ: np.float64 = 0,
                α: np.float64 = .99, s: np.float64 = .01,
                steps: int = 100,
                fit='prf') -> PiecewiseConstantHistory:
        '''infer the μ history given the sfs and η history

        η: η history
        λ: regularization strength
        α: relative penalty on L1 vs L2
        s: step size parameter for proximal gradient descent
        steps: number of proximal gradient descent steps
        fit: fit function, 'prf' for Poisson random field, 'kl' for
             Kullback-Leibler divergence, 'lsq' for least-squares
        '''
        assert λ >= 0, 'λ must be nonnegative'
        assert 0 <= α <= 1, 'α must be in the unit interval'

        D = (np.eye(η.m(), k=0) - np.eye(η.m(), k=-1))
        W = np.eye(η.m())
        W[0, 0] = 0
        D2 = D.T @ W @ D

        # gradient of differentiable piece of loss function
        if fit == 'prf':
            def misfit_func(*args, **kwargs):
                return -self.ℓ(*args, **kwargs)
        elif fit == 'kl':
            misfit_func = self.d_kl
        elif fit == 'lsq':
            misfit_func = self.lsq
        else:
            raise ValueError(f'unrecognized fit argument {fit}')
        def grad_f(z):
            history = JointHistory(η,
                                   PiecewiseConstantHistory(η.change_points, z)
                                   )
            return misfit_func(history, grad=True) \
                   + λ * (1 - α) * D2 @ z
        # initialize using constant μ history MLE
        z = self.constant_μ_MLE(η).vals
        for _ in range(steps):
            g = grad_f(z)
            if not all(np.isfinite(g)):
                raise RuntimeError(f'invalid gradient: {g}')
            z = z - s * g
            if α > 0:
                z = ptv.tv1_1d(z, λ * α)
            z = np.clip(z, 1e-6, np.inf)
            if not all(np.isfinite(z)):
                raise RuntimeError(f'invalid z value: {z}')
        return PiecewiseConstantHistory(η.change_points, z)

    def plot(self, history: JointHistory = None, prf_quantiles=False):
        '''plot the SFS data and optionally the expected SFS under history

        history: joint η and μ history
        prf_quantiles: if True show 95% marginal intervals using the Poisson
                       random field
        '''
        if history is not None:
            ξ = self.ξ(history)
            plt.plot(range(1, self.n), ξ, 'r--', label=r'$\xi$')
            if prf_quantiles:
                ξ_lower = poisson.ppf(.025, ξ)
                ξ_upper = poisson.ppf(.975, ξ)
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
