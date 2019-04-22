#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import binom
from scipy.stats import poisson
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from typing import Tuple


class DemEnt():
    '''A class that implements the model of Rosen et al., but adds a Poisson
    random field for generating the SFS from ξ
    '''

    def __init__(self, n: int, t: np.ndarray, y: np.ndarray, r: float = 1,
                 infinite: bool = True):
        '''
        n: number of sampled haplotypes
        t: The time axis. The last epoch in t extends to infinity in Rosen, but
           we truncate if infinite=False
        y: vector of eta pieces
        r: mutation rate is per genome per generation (controls SFS noise)
        infinite: extend to infinity with constant size (otherwise truncate)
        '''
        if t[0] != 0:
            raise ValueError('the first element of t must be 0')
        if any(np.diff(t) <= 0) or np.isinf(t).sum():
            raise ValueError('t must be monotonically increasing and finite')
        if any(y <= 0) or np.isinf(t).sum():
            raise ValueError('elements of y must be finite and positive')
        if n < 2:
            raise ValueError('n must be larger than 1')
        self.infinite = infinite
        self.n = n
        self.t = t
        self.y_true = y
        self.r = r
        self._binom_array = binom(np.arange(2, n + 1), 2)[:, np.newaxis]
        self.A = DemEnt.A(n)
        self._s = np.diff(self.t)
        self.simulate_sfs()  # this sets self.sfs

    @staticmethod
    def A(n: int) -> np.ndarray:
        '''The A_n matrix of Polanski and Kimmel (2003) (equations 13–15)
        Using notation of Rosen et al. (2018)

        n: number of sampled haplotypes
        '''
        A = np.zeros((n - 1, n - 1))
        b = np.arange(1, n - 1 + 1)
        A[:, 0] = 6 / (n + 1)
        A[:, 1] = 30 * (n - 2 * b) / (n + 1) / (n + 2)
        for col in range(n - 3):
            j = col + 2
            c1 = - (1 + j) * (3 + 2 * j) * (n - j) / j / (2 * j - 1) / (n+j+1)
            c2 = (3 + 2 * j) * (n - 2 * b) / j / (n + j + 1)
            A[:, col + 2] = c1 * A[:, col] + c2 * A[:, col + 1]
        return A

    def c(self, y: np.ndarray = None, jacobian: bool = False) -> np.ndarray:
        '''coalescence vector computed by eqn (3) of Rosen et al. (2018), and
        it's Jacobian

        y: η(t) values
        jacobian: compute jacobian if True
        '''
        if y is None:
            y = self.y_true
        # M_2 from Rosen et al. (2018)
        x = np.exp(- self._s / y)
        x = np.insert(x, 0, 1)
        y_diff = np.insert(np.diff(y), 0, y[0])
        if self.infinite:
            # when using infinite domain, extend last point to infty
            x = np.concatenate((x, [0]))  # final exponential is zero
            y_diff = np.concatenate((y_diff, [0]))  # final diff is zero
        M2 = np.cumprod(x[np.newaxis, :-1] ** self._binom_array, axis=1) \
            / self._binom_array
        c = M2.dot(y_diff)
        if not jacobian:
            return c
        raise NotImplementedError('Jacobian not implemented yet')
        # dM2dy = np.zeros((M2.shape[0], M2.shape[1], k))
        # for depth in range(k):
        #     dM2dy[:, (depth + 1):, depth] = binom_array \
        #       * (self.t[depth + 1] - self.t[depth]) / (y[depth] ** 2) \
        #       * M2[:, (depth + 1):]
        # J = np.tensordot(dM2dy, y_diff, ([1], [0])) \
        #     + M2 @ (np.eye(k) - np.eye(k, k=-1))
        # return c, J

    def xi(self, y: np.ndarray = None) -> np.ndarray:
        '''The expected SFS vector as in Rosen et al., but with an explicit
        mutation rate that controls noise

        y: optional η(t) values, uses self.y_true if None
        '''
        return self.r * self.A.dot(self.c(y))

    def simulate_sfs(self) -> None:
        '''simulate a SFS under the Poisson random field model'''
        self.sfs = poisson.rvs(self.xi())

    def ell(self, y: np.ndarray = None) -> float:
        '''Poisson random field log-likelihood

        y: η(t) values, use self.y_true if None
        '''
        if not hasattr(self, 'sfs'):
            raise ValueError('use simulate_sfs() to generate data first')
        xi = self.xi(y)
        ell = - xi.sum() + self.sfs.dot(np.log(xi))
        return ell

    def constant_MLE(self) -> float:
        '''MLE for constant demography. Updates the instance's self.y_inferred
        based on the constant MLE
        '''
        # number of segregating sites
        S = self.sfs.sum()
        # Harmonic number
        H = (1 / np.arange(1, self.n - 1)).sum()
        # constant MLE
        self.y_inferred = (S / 2 / H / self.r) * np.ones(len(self.t) - 1)

    def tmrca_cdf(self) -> np.ndarray:
        '''return the CDF of the TMRCA of n samples from a population of size
        eta_0, using eqn (3.39) from Wakeley, at each time in self.t.
        '''
        if (np.diff(self.y_inferred) != 0).sum() > 0:
            raise NotImplementedError('tmrca_cdf() only available for constant'
                                      'demographies')
        eta_0 = self.y_inferred[0]
        with np.errstate(divide='ignore'):
            binom_factors = self._binom_array / (self._binom_array
                                                 - self._binom_array.T)
        binom_factors[np.isinf(binom_factors)] = 1
        binom_prod = np.prod(binom_factors, axis=0)

        return 1 - np.exp(-self._binom_array * self.t / eta_0).T @ binom_prod

    def loss(self, y, y_prior, lambda_prior: float = 0,
             lambda_diff: np.ndarray = None) -> float:
        '''negative log likelihood (Poisson random field) and regularizations
        on divergence from a prior and on the derivative

        y: list of η(t) values
        y_prior: list of η(t) values
        lambda_prior: regularization strength on Bregman divergence from prior
        lambda_diff: regularization strength on derivative at each time
        '''
        # generalized KL divergence (a Bregman divergence) from y_prior
        # we need the factor of self._s to encode the Lebesgue measure,
        # especially for a non-regular time grid (i.e. log)
        R_prior = lambda_prior * ((y * np.log(y/y_prior)
                                   - y + y_prior) * self._s).sum()

        # L2 on derivative
        # note the differential self._s cancels with the denominator
        # differential in the derivative dη/dt, so we just use diff(y) ~ dη
        if lambda_diff is None:
            lambda_diff = np.zeros(len(self.t))
        R_diff = (lambda_diff[:-2] * np.diff(y)**2).sum()

        return - self.ell(y) + R_prior + R_diff

    def invert(self, iterates: int = 1, lambda_prior: float = 1,
               lambda_diff_min: float = 1, lambda_diff_max: float = None):
        '''infer the demography given the simulated sfs

        iterates: number of outer iterates
        lambda_prior: initial regularization for prior
        lambda_diff_min: minimum regularization for derivative
        lambda_diff_max: ramp derivative penalty to this value as we approach
                          coalescent horizon (no ramp if None)
        '''
        # Initialize with a MLE constant demography
        self.constant_MLE()
        print('constant MLE initialization')
        self.plot()

        # derivative penalty ramp based on TMRCA CDF
        if lambda_diff_max is None:
            self._lambda_diff = None
        else:
            if lambda_diff_max < lambda_diff_min:
                raise ValueError('lambda_diff_ramp {} must be greater than '
                                 'lambda_diff {}'.format(lambda_diff_max,
                                                         lambda_diff_min))
            self._lambda_diff = (lambda_diff_min
                                 + (lambda_diff_max
                                    - lambda_diff_min) * self.tmrca_cdf())

        # prior set to the initial (constant population MLE)
        y_prior = self.y_inferred

        # (meta-)optimization
        for iterate in range(1, iterates + 1):
            result = minimize(self.loss,
                              y_prior,
                              args=(y_prior, lambda_prior, self._lambda_diff),
                              # jac=gradF,
                              method='L-BFGS-B',
                              options=dict(
                                           # ftol=1e-10,
                                           maxfun=np.inf
                                           ),
                              bounds=[(1, None)] * len(y_prior))
            if not result.success:
                print(result.message)
            else:
                # update inference and prior
                self.y_inferred = result.x
                y_prior = result.x
                print('iteration {}: λ_prior = {}, '
                      'ℓ = {}'.format(iterate,
                                      lambda_prior,
                                      self.ell(self.y_inferred)))
                self.plot()
                # increment regularization strength
                lambda_prior /= 10

    def plot(self) -> None:
        '''plot the true η(t), and optionally a fitted one y if self.y_inferred
        is not None
        '''
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].step(self.t[:-1], self.y_true, 'k', where='post', label='true')
        if hasattr(self, 'y_inferred'):
            axes[0].step(self.t[:-1], self.y_inferred, 'r', where='post',
                         label='inverted')
        if hasattr(self, '_lambda_diff'):
            ax_right = axes[0].twinx()
            ax_right.set_ylabel('$\\lambda_{\\eta\'}$', color='tab:blue')
            ax_right.plot(self.t, self._lambda_diff, 'tab:blue')
            ax_right.tick_params(axis='y', labelcolor='tab:blue')
            ax_right.set_yscale('log')
        axes[0].set_xlabel('$t$')
        axes[0].set_ylabel('$\\eta(t)$')
        axes[0].legend()
        axes[0].legend(loc='upper center')
        axes[0].set_ylim([0, None])
        axes[0].set_xscale('log')
        # axes[0].set_yscale('log')

        if hasattr(self, 'y_inferred'):
            xi = self.xi(self.y_inferred)
            xi_lower = poisson.ppf(.025, xi)
            xi_upper = poisson.ppf(.975, xi)
            axes[1].plot(range(1, len(xi) + 1), xi, 'r--', label=r'$\xi$')
            axes[1].fill_between(range(1, len(xi) + 1),
                                 xi_lower, xi_upper,
                                 facecolor='r', alpha=0.25,
                                 label='inner 95%\nquantile')
        axes[1].plot(range(1, len(self.sfs) + 1), self.sfs,
                     'k.', alpha=.25, label=r'simulated')
        axes[1].set_xlabel('$i$')
        axes[1].set_ylabel(r'$\xi_i$')
        axes[1].set_xscale('log')
        axes[1].set_yscale('symlog')
        axes[1].legend()
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.show()
