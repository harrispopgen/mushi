#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import binom
from scipy.stats import poisson
import matplotlib
from matplotlib import pyplot as plt, rc_context
from matplotlib import patheffects
plt.xkcd()
matplotlib.rcParams['path.effects'] = [patheffects.withStroke(linewidth=0, foreground="w")]

# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

class DemEnt():
    '''
    A class that implements the model of Rosen et al., but adds a Poisson random
    field for generating the SFS from ξ
    '''
    def __init__(self, n: int, t, y, infinite=False):
        '''
        n is number of sampled haplotypes
        The last epoch in t extends to infinity in Rosen, but we truncate instead
        '''
        assert t[0] == 0
        assert not np.isinf(t[-1])
        assert all(np.diff(t) > 0)
        assert all(y > 0)
        assert n > 1
        self.n = n
        self.t = t
        self.y = y
        self.A = _init_A(n)
        self.s = np.diff(self.t)
        self.binom_array = binom(np.arange(2, n + 1), 2)
        self.simulate_sfs()

    def _init_A(n):
        # The A_n matrix of Polanski and Kimmel (2003) (equations 13–15)
        # Using notation of Rosen et al. (2018)
        A = np.zeros((n - 1, n - 1))
        b = np.arange(1, n - 1 + 1)
        A[:, 0] = 6 / (n + 1)
        A[:, 1] = 30 * (n - 2 * b) / (n + 1) / (n + 2)
        for col in range(n - 3):
            j = col + 2
            c1 = - (1 + j) * (3 + 2 * j) * (n - j) / j / (2 * j - 1) / (n + j + 1)
            c2 = (3 + 2 * j) * (n - 2 * b) / j / (n + j + 1)
            A[:, col + 2] = c1 * A[:, col] + c2 * A[:, col + 1]
        return A
        
    def c(self, y, jacobian=False):
        '''
        coalescence vector computed by eqn (3) of Rosen et al. (2018), and it's jacobian
        input y is vector of constant pieces for the demography
        '''
        # M_2 from Rosen et al. (2018)
        k = len(y)
        x = np.exp(- self.s / y)
        x = np.insert(x, 0, 1)
        M2 = np.tile(np.array([1 / self.binom_array]).transpose(), (1, k)) * np.cumprod((x[np.newaxis, :-1] ** self.binom_array[:, np.newaxis]), axis=1)
        y_diff = np.insert(np.diff(y), 0, y[0])
        c = M2.dot(y_diff)
        if not jacobian:
            return c
        raise NotImplementedError('Jacobian not implemented yet')
        dM2dy = np.zeros((M2.shape[0], M2.shape[1], k))
        for depth in range(k):
            dM2dy[:, (depth + 1):, depth] = self.binom_array[:, np.newaxis] * (self.t[depth + 1] - self.t[depth]) / (y[depth] ** 2) * M2[:, (depth + 1):]
        J = np.tensordot(dM2dy, y_diff, ([1], [0])) + M2 @ (np.eye(k) - np.eye(k, k=-1))
        return c, J
    # def c(self, y, jacobian=False):
    #     '''
    #     coalescence vector computed by trapezoidal quadrature on eqn (1) from Rosen et al.
    #     '''
    #     # M_2 from Rosen et al. (2018)
    #     k = len(y)
    #     x = (y[1:] / y[:-1] - 1)**(-self.s / np.diff(y))
    #     print(y)
    #     print(x)
    #     # x = np.insert(x, 0, 1)
    #
    #     c = np.ones(self.n - 1)
    #     for m in range(2, len(c) + 2):
    #         m_choose_2 = self.binom_array[m - 2]
    #         c[m - 2] = (m_choose_2 / 2) * (self.s * np.cumprod(x ** m_choose_2) * (1 / y[:-1] + x ** m_choose_2 / y[1:])).sum()
    #     print(c)
    #
    #     if not jacobian:
    #         return c
    #     else:
    #         raise NotImplementedError()

    def xi(self, y):
        xi = self.A.dot(self.c(y))
        return xi

    def simulate_sfs(self, plot=False):
        '''
        simulate a SFS under the Poisson random field model
        '''
        xi = self.xi(self.y)
        self.sfs = poisson.rvs(xi)
        if plot:
            self.plot()

    def ell(self, y):
        '''
        Poisson random field log-likelihood
        '''
        assert self.sfs is not None
        xi = self.xi(y)
        ell = - xi.sum() + self.sfs.dot(np.log(xi))
        # ell = -((xi - self.sfs) ** 2).sum()
        return ell

    def plot(self, y=None, y_label=None):
        '''
        plot the true η(t), and optionally a fitted y
        '''
        assert self.y is not None
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].step(self.t[:-1], self.y,
                     'k', where='post', label='true')
        if y is not None:
            axes[0].step(self.t[:-1], y,
                         'r', where='post', label=y_label)
        axes[0].set_xlabel('$t$')
        axes[0].set_ylabel('$\eta(t)$')
        axes[0].legend()
        axes[0].legend(loc=(1.04,.5))
        axes[0].set_ylim([0, None])
        # axes[0].set_xscale('symlog')
        # axes[0].set_yscale('log')

        if y is not None:
            xi = self.xi(y)
            xi_lower = poisson.ppf(.025, xi)
            xi_upper = poisson.ppf(.975, xi)
            axes[1].plot(range(1, len(xi) + 1), xi, 'r--', label=r'$\xi$')
            axes[1].fill_between(range(1, len(xi) + 1), xi_lower, xi_upper,
                             facecolor='r', alpha=0.25,
                             label='inner 95%\nquantile')
        axes[1].plot(range(1, len(self.sfs) + 1), self.sfs,
                     'k.', alpha=.5, markersize=2,
                     label=r'simulated')
        axes[1].set_xlabel('$i$')
        axes[1].set_ylabel(r'$\xi_i$')
        axes[1].set_xscale('log')
        axes[1].set_yscale('symlog')
        axes[1].legend()
        axes[1].legend(loc=(1.04,.5))

        plt.tight_layout()
        plt.show()
