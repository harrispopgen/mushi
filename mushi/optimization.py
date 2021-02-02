r"""Optimizer objects

"""

import abc
import numpy as np
from typing import Callable, Tuple
import prox_tv as ptv
from scipy.linalg import cholesky_banded, cho_solve_banded
from functools import lru_cache


class Optimizer(metaclass=abc.ABCMeta):
    """Abstract base class for optimizers

    Attributes:
        x: solution point

    Args:
        verbose: flag to print convergence messages
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.x = None

    @abc.abstractmethod
    def initialize(self, x: np.ndarray) -> None:
        """initialize solution point x, and any auxiliary variables"""
        pass

    def check_x(self):
        """test if x is defined"""
        if self.x is None:
            raise TypeError('solution point x is not initialized')

    @abc.abstractmethod
    def f(self) -> np.float64:
        """evaluate cost function at current solution point"""
        pass

    @abc.abstractmethod
    def step(self) -> None:
        """take an optimization step and update solution point"""
        pass

    def run(self, x: np.ndarray,
            tol: np.float64 = 1e-6, max_iter: int = 100) -> np.ndarray:
        """Optimize until convergence criteria are met

        Args:
            x: initial point
            tol: relative tolerance in objective function
            max_iter: maximum number of iterations

        Returns:
            x: solution point
        """
        self.initialize(x)
        self.check_x()
        # initial objective value
        f = self.f()
        if self.verbose:
            print(f'initial objective {f:.6e}', flush=True)
        k = 0
        for k in range(1, max_iter + 1):
            self.step()
            if not np.all(np.isfinite(self.x)):
                print('warning: x contains invalid values', flush=True)
            # terminate if objective function is constant within tolerance
            f_old = f
            f = self.f()
            rel_change = np.abs((f - f_old) / f_old)
            if self.verbose:
                print(f'iteration {k}, objective {f:.3e}, '
                      f'relative change {rel_change:.3e}',
                      end='        \r', flush=True)
            if rel_change < tol:
                if self.verbose:
                    print('\nrelative change in objective function '
                          f'{rel_change:.2g} '
                          f'is within tolerance {tol} after {k} iterations',
                          flush=True)
                return self.x
        if self.verbose and k > 0:
            print(f'\nmaximum iteration {max_iter} reached with relative '
                  f'change in objective function {rel_change:.2g}',
                  flush=True)
        return self.x


class LineSearcher(Optimizer):
    """Abstract class for an optimizer with Armijo line search

    Args:
        s0: initial step size
        max_line_iter: maximum number of line search steps
        gamma: step size shrinkage rate for line search
        verbose: flag to print convergence messages
    """

    def __init__(self, s0: np.float64 = 1, max_line_iter: int = 100,
                 gamma: np.float64 = 0.8, verbose: bool = False):
        self.s0 = s0
        self.max_line_iter = max_line_iter
        self.gamma = gamma
        super().__init__(verbose=verbose)


class AccProxGrad(LineSearcher):
    r"""Nesterov accelerated proximal gradient method with backtracking line
    search [1]_.

    The optimization problem solved is:

    .. math::
        \arg\min_x g(x) + h(x)

    where :math:`g` is differentiable, and the proximal operator for :math:`h`
    is available.

    Args:
        g: differentiable term in objective function
        grad_g: gradient of g
        h: non-differentiable term in objective function
        prox: proximal operator corresponding to h
        verbose: flag to print convergence messages
        line_search_kwargs: line search keyword arguments,
                            see :py:class:`mushi.optimization.LineSearcher`

    References
    ----------
    .. [1]
    https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

    """

    def __init__(self,
                 g: Callable[[np.ndarray], np.float64],
                 grad_g: Callable[[np.ndarray], np.float64],
                 h: Callable[[np.ndarray], np.float64],
                 prox: Callable[[np.ndarray, np.float64], np.float64],
                 verbose: bool = False,
                 **line_search_kwargs):
        self.g = g
        self.grad_g = grad_g
        self.h = h
        self.prox = prox
        super().__init__(verbose=verbose, **line_search_kwargs)

    def f(self):
        self.check_x()
        return self.g(self.x) + self.h(self.x)

    def initialize(self, x: np.ndarray) -> None:
        # initialize solution point
        self.x = x
        # initialize momentum iterate
        self.q = self.x
        # initialize step size
        self.s = self.s0
        # initialize step counter
        self.k = 0

    def step(self) -> None:
        """step with backtracking line search"""
        self.check_x()
        # evaluate differtiable part of objective at momentum point
        g1 = self.g(self.q)
        grad_g1 = self.grad_g(self.q)
        if not np.all(np.isfinite(grad_g1)):
            raise RuntimeError(f'invalid gradient:\n{grad_g1}')
        # store old iterate
        x_old = self.x
        # Armijo line search
        for line_iter in range(self.max_line_iter):
            # new point via prox-gradient of momentum point
            self.x = self.prox(self.q - self.s * grad_g1, self.s)
            # G_s(q) as in the notes linked above
            G = (1 / self.s) * (self.q - self.x)
            # test g(q - sG_s(q)) for sufficient decrease
            if self.g(self.q - self.s * G) <= (
                                            g1 - self.s * (grad_g1 * G).sum()
                                            + (self.s / 2) * (G ** 2).sum()):
                # Armijo satisfied
                break
            else:
                # Armijo not satisfied
                self.s *= self.gamma  # shrink step size

        # update step count
        self.k += 1
        # update momentum point
        self.q = self.x + ((self.k - 1) / (self.k + 2)) * (self.x - x_old)

        if line_iter == self.max_line_iter - 1:
            print('warning: line search failed', flush=True)
            # reset step size
            self.s = self.s0


class ThreeOpProxGrad(AccProxGrad):
    r"""Three operator splitting proximal gradient method with backtracking
    line search [2]_.

    The optimization problem solved is:

    .. math::
        \arg\min_x g(x) + h_1(x) + h_2(x)

    where :math:`g` is differentiable, and the proximal operators for
    :math:`h_1` and :math:`h_2` are available.

    Args:
        g: differentiable term in objective function
        grad_g: gradient of g
        h1: 1st non-differentiable term in objective function
        h2: 2nd non-differentiable term in objective function
        prox1: proximal operator corresponding to h1
        prox2: proximal operator corresponding to h2
        verbose: print convergence messages
        line_search_kwargs: line search keyword arguments,
                            see :py:class:`mushi.optimization.LineSearcher`

    References:
        .. [2] Pedregosa, Gidel, Adaptive Three Operator Splitting in
               Proceedings of the 35th International Conference on Machine
               Learning, Proceedings of Machine Learning Research., J. Dy, A.
               Krause, Eds. (PMLR, 2018), pp. 4085–4094.

    """

    def __init__(self,
                 g: Callable[[np.ndarray], np.float64],
                 grad_g: Callable[[np.ndarray], np.float64],
                 h1: Callable[[np.ndarray], np.float64],
                 prox1: Callable[[np.ndarray, np.float64], np.float64],
                 h2: Callable[[np.ndarray], np.float64],
                 prox2: Callable[[np.ndarray, np.float64], np.float64],
                 verbose: bool = False,
                 **line_search_kwargs):
        super().__init__(g, grad_g, h1, prox1, verbose=verbose,
                         **line_search_kwargs)
        self.h2 = h2
        self.prox2 = prox2

    def f(self):
        return super().f() + self.h2(self.x)

    def initialize(self, x: np.ndarray) -> None:
        super().initialize(x)
        # dual variable
        self.u = np.zeros_like(self.q)

    def step(self) -> None:
        """step with backtracking line search"""
        self.check_x()
        # evaluate differentiable part of objective
        g1 = self.g(self.q)
        grad_g1 = self.grad_g(self.q)
        if not np.all(np.isfinite(grad_g1)):
            raise RuntimeError(f'invalid gradient:\n{grad_g1}')
        # Armijo line search
        for line_iter in range(self.max_line_iter):
            # new point via prox-gradient of momentum point
            self.x = self.prox(self.q - self.s * (self.u + grad_g1), self.s)
            # quadratic approximation of objective
            Q = (g1 + (grad_g1 * (self.x - self.q)).sum()
                    + ((self.x - self.q) ** 2).sum() / (2 * self.s))
            if self.g(self.x) - Q <= 0:
                # sufficient decrease satisfied
                break
            else:
                # sufficient decrease not satisfied
                self.s *= self.gamma  # shrink step size
        if line_iter == self.max_line_iter - 1:
            print('warning: line search failed', flush=True)
            # reset step size
            self.s = self.s0

        # update z variables with 2nd prox
        self.q = self.prox2(self.x + self.s * self.u, self.s)
        # update u variables: dual variables
        self.u = self.u + (self.x - self.q) / self.s
        # grow step size
        self.s = min(self.s / self.gamma ** 2, self.s0)


class TrendFilter(Optimizer):
    r"""Mixed trend filtering via specialized ADMM as in [3]_, section 5.2.

    The optimization problem solved is:

    .. math::
        \arg\min_{\beta \in \mathbb{R}} \frac{1}{2} \|y - \beta\|_2^2
        + \sum_{\ell=1}^r \lambda_\ell \| D^{k_\ell + 1} \beta \|_1

    where :math:`r` is the number of elements of k.

    Args:
        ks: tuple of integer trend filter orders
        lambdas: tuple of penalties corresponding to each k
        rhos: ADMM convergence parameter
        verbose: print convergence messages

    References:
        .. [3] Aaditya Ramdas and Ryan J. Tibshirani.
               Fast and flexible admm algorithms for trend filtering.
               Journal of Computational and Graphical Statistics
               25.3 (2016): 839-858.
    """

    def __init__(self, ks: Tuple[int], lambdas: Tuple[np.float64],
                 rhos: Tuple[np.float64] = None, verbose: bool = False):
        self.k = ks
        self.λ = lambdas
        self.r = len(self.k)
        if rhos is None:
            # default ADMM parameters used in cited paper
            self.ρ = self.λ
        else:
            self.ρ = rhos
        if len(self.λ) != self.r:
            raise ValueError('ks and lambdas must have equal length')
        if len(self.ρ) != self.r:
            raise ValueError('ks and rhos must have equal length')
        super().__init__(verbose=verbose)

    def f(self):
        self.check_x()
        fit = 0.5 * np.sum((self.x - self.y) ** 2)
        penalty = sum(λ * np.linalg.norm(np.diff(self.x, k + 1, axis=0), 1)
                      for k, λ in zip(self.k, self.λ))
        return fit + penalty

    def initialize(self, x: np.ndarray) -> None:
        # input signal
        self.y = x
        if self.y.ndim == 1:
            self.y = self.y[:, np.newaxis]
            # remember the signal was 1D
            self.oneD = True
        else:
            self.oneD = False
        self.n = self.y.shape[0]
        # initialize solution point (β in the paper cited)
        self.x = np.zeros_like(self.y)
        # initialize Lagrangian dual variables
        self.α = [np.zeros_like(self.y)[:(self.n - k)] for k in self.k]
        self.u = [np.zeros_like(self.y)[:(self.n - k)] for k in self.k]

        self.D, self.DTD = self._D_DTD(self.n, self.k)
        self.c = self._choleskify(self.n, self.k, self.ρ)

    def step(self):
        self.x = cho_solve_banded(
            (self.c, False),
            self.y + sum(self.ρ[i] * self.D[i].T @ (self.α[i] + self.u[i])
                         for i in range(self.r)),
            check_finite=False)
        for i in range(self.r):
            Dx = self.D[i] @ self.x
            for j in range(self.x.shape[1]):
                self.α[i][:, j] = ptv.tv1_1d(Dx[:, j] - self.u[i][:, j],
                                             self.λ[i] / self.ρ[i])
            self.u[i] += self.α[i] - Dx

    def run(self, *args, **kwargs) -> np.ndarray:
        solution = super().run(*args, **kwargs)
        # squeeze out singleton dimension if input was a 1D
        if self.oneD:
            return np.squeeze(solution)
        # else:
        return solution

    @staticmethod
    @lru_cache()
    def _D_DTD(n: int, k: Tuple[int]) -> Tuple[Tuple[np.ndarray],
                                               Tuple[np.ndarray]]:
        """difference operator for each order in k

        Args:
            n: number of points in signal
            k: tuple of integer trend filter orders

        Returns:
            D: tuple of difference operator for each k
            DTD: D.T @ D corresponding to above
        """
        D = np.eye(n, k=0) - np.eye(n, k=-1)
        D = tuple(np.linalg.matrix_power(D, k)[k:] for k in k)

        return tuple(zip(*((D[i], D[i].T @ D[i]) for i in range(len(k)))))

    @staticmethod
    @lru_cache(maxsize=1024)
    def _choleskify(n: int, k: Tuple[int], ρ: Tuple[np.float64]) -> np.ndarray:
        """cholesky decomposition needed for linear solves in trend estimate

        Args:
            n: number of points in signal
            k: tuple of integer trend filter orders
            ρ: ADMM parameter for each order in k

        Returns:
            Cholesky decomposition
        """
        DTD = TrendFilter._D_DTD(n, k)[1]
        A = np.eye(n) + sum(ρ[i] * DTD[i] for i in range(len(k)))
        # A is a banded Hermitian positive definite matrix with upper/lower
        # bandwidth bw. Express in upper diagonal ordered form
        bw = max(k)
        Ab = np.zeros((bw + 1, A.shape[1]))
        for u in range(bw + 1):
            Ab[-(1 + u), u:] = np.diag(A, k=u)

        return cholesky_banded(Ab, check_finite=False)
