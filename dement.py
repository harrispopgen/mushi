#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import binom
from scipy.stats import poisson
from scipy.optimize import minimize, check_grad
import matplotlib as mplt
from matplotlib import pyplot as plt
mplt.rc('text', usetex=True)
mplt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

class DemEnt():
    def __init__(self, n: int, t):
        '''
        n is number of sampled haplotypes
        The last epoch in t extends to infinity
        '''
        assert t[0] == 0
        assert t[-1] == np.inf
        assert n > 1
        self.n = n
        self.t = t
        # The A_n matrix of Polanski and Kimmel (2003) (equations 13–15)
        # Using notation of Rosen et al. (2018)
        self.A = np.zeros((n - 1, n - 1))
        b = np.arange(1, n - 1 + 1)
        self.A[:, 0] = 6 / (n + 1)
        self.A[:, 1] = 30 * (n - 2 * b) / (n + 1) / (n + 2)
        for col in range(n - 3):
            j = col + 2
            c1 = - (1 + j) * (3 + 2 * j) * (n - j) / j / (2 * j - 1) / (n + j + 1)
            c2 = (3 + 2 * j) * (n - 2 * b) / j / (n + j + 1)
            self.A[:, col + 2] = c1 * self.A[:, col] + c2 * self.A[:, col + 1]
        self.s = np.diff(self.t)
        self.binom_array = binom(np.arange(2, n + 1), 2)
        self.sfs = None
        self.y = None

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
        dM2dy = np.zeros((M2.shape[0], M2.shape[1], k))
        for depth in range(k):
            dM2dy[:, (depth + 1):, depth] = self.binom_array[:, np.newaxis] * (self.t[depth + 1] - self.t[depth]) / (y[depth] ** 2) * M2[:, (depth + 1):]
        J = np.tensordot(dM2dy, y_diff, ([1], [0])) + M2 @ (np.eye(k) - np.eye(k, k=-1))
        return c, J

    def xi(self, y):
        xi = self.A.dot(self.c(y))
        return xi


    def simulate_sfs(self, y):
        '''
        simulate a SFS under the Poisson random field model
        '''
        xi = self.xi(y)
        self.sfs = poisson.rvs(xi)
        self.y = y

    def ell(self, y):
        '''
        Poisson random field log-likelihood, and its gradient
        '''
        assert self.sfs is not None
        xi = self.xi(y)
        ell = - xi.sum() + self.sfs.dot(np.log(xi))
        # ell = -((xi - self.sfs) ** 2).sum()
        return ell

    @staticmethod
    def D(y, y_prime):
        '''
        generalized KL divergence (a Bregman divergence)
        '''
        # NOTE: need to do fancy proj stuff when penalty is not differentiable!
        # return (y * np.log(y/y_prime) - y + y_prime).sum()
        # return ((y - y_prime) ** 2).sum()
        # return (y * np.log(y/y_prime) - y + y_prime).sum() + 10 *(np.diff(y)**2).sum()
        # return (y**2).sum()
        # L1 on diff seems natural, it will make runs of zero diff, like a bottleneck?
        d = np.abs(np.diff(y)**2).sum()
        d += 0.001 * ((y - y_prime) ** 2).sum()
        # d += (y[-1] - y[-2]) ** 2
        return d

    def ell_tilde(self, y, y_prime, lambda_: float):
        '''
        penalized log likelihood
        '''
        return self.ell(y) - lambda_ * self.D(y, y_prime)

    def plot(self, y=None):
        assert self.y is not None
        plt.subplot(1, 2, 1)
        plt.step(self.t[:-1], self.y, 'r', where='post', label='true', alpha=.5)
        plt.plot([self.t[-2], 1.5 * self.t[-2]], [self.y[-1], self.y[-1]], 'r', alpha=.5)
        if y is not None:
            plt.step(self.t[:-1], y, 'k', where='post', label='inferred', alpha=.5)
            plt.plot([self.t[-2], 1.5 * self.t[-2]], [y[-1], y[-1]], 'k', alpha=.5)
        plt.xlabel('$t$')
        plt.ylabel('$\eta(t)$')
        plt.legend()
        # plt.yscale('log')
        plt.xlim([0, 1.5 * self.t[-2]])
        plt.ylim([0, None])
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.sfs) + 1), self.sfs, 'r.', alpha=.5,
                 label=r'simulated sfs')
        if y is not None:
            xi = self.xi(y)
            xi_lower = poisson.ppf(.025, xi)
            xi_upper = poisson.ppf(.975, xi)
            plt.plot(range(1, len(xi) + 1), xi, 'k--', label=r'$\boldsymbol{\xi}$')
            plt.fill_between(range(1, len(xi) + 1), xi_lower, xi_upper,
                             facecolor='k', alpha=0.25,
                             label='inner 95% quantile')
        plt.xlabel('$i$')
        plt.ylabel(r'$\xi_i$')
        plt.xscale('log')
        plt.yscale('symlog')
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    """
    usage: python dement.py -h
    """
    import argparse

    # parser = argparse.ArgumentParser(description='infer projection test given'
    #                                              ' list of test genome genotype'
    #                                              ' likelihoods and a list of'
    #                                              ' reference population site'
    #                                              ' frequencies')
    # parser.add_argument('metadata',
    #                     type=str,
    #                     help='path to sample metadata excel spreadsheet')
    #
    #
    # args = parser.parse_args()

    t = np.array(list(np.linspace(0, 400, 200)) + [np.inf])
    y_true = 1000 * np.ones(len(t) - 1)# + 50 * np.exp((500 - t[:-1]) / 500)
    y_true[:50] = y_true[49] * np.exp((t[49] - t[:50]) / 1)
    y_true[100:] = 2000
    # y_true += 20*np.random.randn(len(y_true))
    y_true = 100 * (10 * np.exp(-t[:-1]/15) + 1 + 5 * np.array(t[:-1] > 200, float))
    n = 200
    dement = DemEnt(n, t)
    dement.simulate_sfs(y_true)
    dement.plot()

    lambda_ = 1e-4

    # We initialize by fitting a constant population size.
    # According to my scribbles, the MLE of N assuming eta(t) = N is
    # $\hat N = \frac{S}{4 H_{n-1}}$, where S is the number of segregating sites
    # (the sum of the observed s vector) and H_{n-1} is the nth harmonic number
    S = dement.sfs.sum()
    H = (1 / np.arange(1, len(dement.sfs))).sum()
    y_prime = (S / 4 / H) * np.ones(len(t) - 1)
    y = y_prime

    F = lambda y: -dement.ell_tilde(y, y_prime, lambda_)

    # F = lambda y: - (y - y_prime - y * np.log(y / y_prime)).sum() + lambda_ * ((xi - A.dot(generate_c(n, y, t)))**2).sum()
    # gradF = lambda y: np.log(y/y_prime) + 2 * lambda_ * (generate_dcdy(n, y, t).T @ A.T).dot(xi - A.dot(generate_c(n, y, t)))
    # assert check_grad(F, gradF, y) < 1e-4
    result = minimize(F,
                      y,
                      # jac=gradF,
                      method='L-BFGS-B',
                      options=dict(ftol=1e-10, maxfun=100000),#np.inf),
                      bounds=[(1e-6, None)] * len(y))
    print(result)
    y = result.x

    dement.plot(y)


    #
    # # for i in range(100):
    # #     print(i)
    # #     y = y_prime * np.exp(-lambda_ * (generate_dcdy(n, y, t).T @ A.T).dot(xi - A.dot(generate_c(n, y, t))))
    #
    # xi_fit = A.dot(generate_c(n, y, t))
    # xi_hat_fit = xi_fit / xi_fit.sum()
    #




if __name__  ==  '__main__':
    main()
