#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import binom
from scipy.stats import poisson
from matplotlib import pyplot as plt
from scipy.special import polygamma
from scipy.optimize import minimize
from typing import Tuple


class DemEnt():
    '''
    A class that implements the model of Rosen et al., but adds a Poisson
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
        assert t[0] == 0
        assert not np.isinf(t[-1])
        assert all(np.diff(t) > 0)
        assert all(y > 0)
        assert n > 1
        self.infinite = infinite
        self.n = n
        self.t = t
        self.y_true = y
        self.r = r
        self._binom_array = binom(np.arange(2, n + 1), 2)
        self.A = DemEnt.A(n)
        self._s = np.diff(self.t)
        self.simulate_sfs()  # this sets self.sfs

    @staticmethod
    def A(n: int) -> np.ndarray:
        '''
        The A_n matrix of Polanski and Kimmel (2003) (equations 13–15)
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
        '''
        coalescence vector computed by eqn (3) of Rosen et al. (2018), and it's
        Jacobian

        y: η(t) values
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
        k = len(y_diff)
        M2 = np.tile(np.array([1 / self._binom_array]).transpose(), (1, k)) * \
             np.cumprod((x[np.newaxis, :-1] ** self._binom_array[:, np.newaxis]),
                        axis=1)
        c = M2.dot(y_diff)
        if not jacobian:
            return c
        raise NotImplementedError('Jacobian not implemented yet')
        # dM2dy = np.zeros((M2.shape[0], M2.shape[1], k))
        # for depth in range(k):
        #     dM2dy[:, (depth + 1):, depth] = binom_array[:, np.newaxis] \
        #       * (self.t[depth + 1] - self.t[depth]) / (y[depth] ** 2) * M2[:, (depth + 1):]
        # J = np.tensordot(dM2dy, y_diff, ([1], [0])) + M2 @ (np.eye(k) - np.eye(k, k=-1))
        # return c, J

    def xi(self, y: np.ndarray = None) -> np.ndarray:
        '''
        The expected SFS vector as in Rosen et al., but with an explicit
        mutation rate that controls noise

        y: optional η(t) values, uses self.y_true if None
        '''
        return self.r * self.A.dot(self.c(y))

    def simulate_sfs(self) -> None:
        '''
        simulate a SFS under the Poisson random field model
        '''
        self.sfs = poisson.rvs(self.xi())

    def ell(self, y: np.ndarray = None) -> float:
        '''
        Poisson random field log-likelihood

        y: η(t) values, use self.y_true if None
        '''
        # we must have previously run the simulate_sfs method
        assert hasattr(self, 'sfs')
        xi = self.xi(y)
        ell = - xi.sum() + self.sfs.dot(np.log(xi))
        return ell

    @staticmethod
    def constant_MLE(n: int, S: int, r: float) -> float:
        '''
        MLE for constant demography (see TeX)

        n: number of sampled haplotypes
        S: number of segregating sites (sum of SFS entries)
        r: mutation rate per genome per generation
        '''
        H = (1 / np.arange(1, n - 1)).sum()
        return (S / 2 / H / r)

    @staticmethod
    def coalescent_horizon(n: int, eta_0: float) -> Tuple[float, float]:
        '''
        return the expectation and variance of the TMRCA of n samples from
        a population of size eta_0
        '''
        T_exp = 2 * eta_0 * (1 - 1 / n)
        T_var = 4 * (3 + n * (6 + n * (-9 + np.pi**2))
                     - 6 * n ** 2 * polygamma(1, n)) \
                  * eta_0 ** 2 / (3 * n ** 2)
        return T_exp, T_var

    def loss(self, y, y_prior, lambda_prior: float = 0, lambda_diff: float = 0,
             weights: np.ndarray = None) -> float:
        '''
        negative log likelihood (Poisson random field) and regularizations on
        divergence from a prior and on the derivative

        y: list of η(t) values
        y_prior: list of η(t) values
        lambda_prior: regularization strength on Bregman divergence from prior
        lambda_diff: regularization strength on derivative
        weights: optional weights adjusting derivative penalization
        '''
        # Lebesgue measure on the modeled time interval
        # This may seem pedantic, but it's nicely robust to a non-regular time
        # grid (i.e. np.logspace)
        dmu = np.diff(self.t)

        # gaussian transformed measure (see TeX)
        # tau = 100 * dement.t[-2]
        # dmu = np.diff(erf(dement.t / tau / np.sqrt(2)))

        # generalized KL divergence (a Bregman divergence) from prior y_prior
        R_prior = ((y * np.log(y/y_prior) - y + y_prior) * dmu).sum()

        # L2 on derivative
        if weights is None:
            weights = np.ones(len(y) - 1)
        R_diff = (weights * np.diff(y)**2 * dmu[:-1]).sum()
    #     return - dement.ell(y) + lambda_ * (R_prior + 1e-1 * R_diff)
        # NOTE: this one fixes diff penalty
        return - self.ell(y) + lambda_prior * R_prior + lambda_diff * R_diff

    def invert(self, iterates: int = 1, lambda_prior: float = 1,
               lambda_diff: float = 1, weight_ramp: bool = False):
        '''
        infer the demography given the simulated sfs

        lambda_prior: initial regularization for prior
        lambda_diff: regularization for derivative
        iterates: number of outer iterates
        weight_ramp: ramp derivative penalty as we approach coalescent horizon
        '''
        # Initialize with a MLE constant demography
        eta_0 = DemEnt.constant_MLE(self.n, self.sfs.sum(), self.r)
        y_constant = eta_0 * np.ones(len(self.t) - 1)
        self.y_inferred = y_constant

        if weight_ramp:
            # approximate the horizon zone under the constant fit (see TeX)
            T_exp, T_var = DemEnt.coalescent_horizon(self.n, eta_0)
            T_sigma = np.sqrt(T_var)
            # time indices where we're within +/- 2 sigma of the expectation
            horizon_zone = np.flatnonzero((T_exp - 2 * T_sigma < self.t[:-1])
                                          & (self.t[:-1] < T_exp + 2 * T_sigma))
            # weights based on coalescent horizon
            # ramp up to a maximum at exp + 2 * sigma
            # we want one for each element of np.diff(y)
            weights = np.ones(len(self.t) - 2)
            max_weight = 1000
            weights[horizon_zone[:-1]] += np.linspace(0, max_weight, len(horizon_zone) - 1)
            weights[horizon_zone[-2]:] = max_weight
        else:
            weights = None

        # prior set to the constant population MLE
        y_prior = y_constant

        for _ in range(iterates):
            result = minimize(self.loss,
                              y_prior,
                              args=(y_prior, lambda_prior, lambda_diff,
                                    weights),
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
                self.plot()
                # increment regularization strength
                lambda_prior /= 10

    def plot(self) -> None:
        '''
        plot the true η(t), and optionally a fitted one y if self.y_inferred is
        not None

        label: optional plot label for self.y_inferred
        '''
        # plt.xkcd()
        # matplotlib.rcParams['path.effects'] = [patheffects.withStroke(linewidth=0,
        #                                                               foreground="w")]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].step(self.t[:-1], self.y_true,
                     'k', where='post', label='true')
        if hasattr(self, 'y_inferred'):
            axes[0].step(self.t[:-1], self.y_inferred,
                         'r', where='post', label='inverted')
        axes[0].set_xlabel('$t$')
        axes[0].set_ylabel('$\\eta(t)$')
        axes[0].legend()
        axes[0].legend(loc=(1.04, 0.5))
        axes[0].set_ylim([0, None])
        # self.axes[0].set_xscale('symlog')
        # self.axes[0].set_yscale('log')

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
        axes[1].legend(loc=(1.04, 0.5))

        plt.tight_layout()
        plt.show()
