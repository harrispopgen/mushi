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
    def _initialize(self, x: np.ndarray) -> None:
        """Initialize solution point x, and any auxiliary variables"""
        pass

    def _check_x(self) -> None:
        """Test if x is defined"""
        if self.x is None:
            raise TypeError('solution point x is not initialized')

    @abc.abstractmethod
    def f(self) -> np.float64:
        """Evaluate cost function at current solution point
        """
        pass

    @abc.abstractmethod
    def _step(self) -> None:
        """Take an optimization step and update solution point"""
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
        self._initialize(x)
        self._check_x()
        # initial objective value
        f = self.f()
        if self.verbose:
            print(f'initial objective {f:.6e}', flush=True)
        k = 0
        for k in range(1, max_iter + 1):
            self._step()
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
                return np.squeeze(self.x)
        if self.verbose and k > 0:
            print(f'\nmaximum iteration {max_iter} reached with relative '
                  f'change in objective function {rel_change:.2g}',
                  flush=True)
        return np.squeeze(self.x)


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
        grad: gradient of g
        h: non-differentiable term in objective function
        prox: proximal operator corresponding to h
        verbose: flag to print convergence messages
        line_search_kwargs: line search keyword arguments,
                            see :py:class:`mushi.optimization.LineSearcher`

    References:

        .. [1]
        https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

    Examples:

        >>> import mushi.optimization as opt
        >>> import numpy as np

        We'll use a squared loss term and a Lasso term for this example, since
        we can specify the solution analytically (indeed, it is directly the
        prox of the Lasso term).

        Define :math:`g(x)` and :math:`\boldsymbol\nabla g(x)`:

        >>> def g(x):
        ...     return 0.5 * np.sum(x ** 2)

        >>> def grad(x):
        ...     return x

        Define :math:`h(x)` and :math:`\mathrm{prox}_h(u)` (we will use a Lasso
        term and the corresponding soft thresholding operator):

        >>> def h(x):
        ...     return np.linalg.norm(x, 1)

        >>> def prox(u, s):
        ...     return np.sign(u) * np.clip(np.abs(u) - s, 0, None)

        Initialize optimizer and define initial point

        >>> fista = opt.AccProxGrad(g, grad, h, prox)
        >>> x = np.zeros(2)

        Run optimization

        >>> fista.run(x)
        array([0., 0.])

        Evaluate cost at the solution point

        >>> fista.f()
        0.0

    """

    def __init__(self,
                 g: Callable[[np.ndarray], np.float64],
                 grad: Callable[[np.ndarray], np.float64],
                 h: Callable[[np.ndarray], np.float64],
                 prox: Callable[[np.ndarray, np.float64], np.float64],
                 verbose: bool = False,
                 **line_search_kwargs):
        self.g = g
        self.grad = grad
        self.h = h
        self.prox = prox
        super().__init__(verbose=verbose, **line_search_kwargs)

    def f(self):
        self._check_x()
        return self.g(self.x) + self.h(self.x)

    def _initialize(self, x: np.ndarray) -> None:
        # initialize solution point
        self.x = x
        # initialize momentum iterate
        self.q = self.x
        # initialize step size
        self.s = self.s0
        # initialize step counter
        self.k = 0

    def _step(self) -> None:
        """step with backtracking line search"""
        self._check_x()
        # evaluate differtiable part of objective at momentum point
        g1 = self.g(self.q)
        grad1 = self.grad(self.q)
        if not np.all(np.isfinite(grad1)):
            raise RuntimeError(f'invalid gradient:\n{grad1}')
        # store old iterate
        x_old = self.x
        # Armijo line search
        for line_iter in range(self.max_line_iter):
            # new point via prox-gradient of momentum point
            self.x = self.prox(self.q - self.s * grad1, self.s)
            # G_s(q) as in the notes linked above
            G = (1 / self.s) * (self.q - self.x)
            # test g(q - sG_s(q)) for sufficient decrease
            if self.g(self.q - self.s * G) <= (
                                            g1 - self.s * (grad1 * G).sum()
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
        grad: gradient of g
        h1: 1st non-differentiable term in objective function
        prox1: proximal operator corresponding to h1
        h2: 2nd non-differentiable term in objective function
        prox2: proximal operator corresponding to h2
        verbose: print convergence messages
        line_search_kwargs: line search keyword arguments,
                            see :py:class:`mushi.optimization.LineSearcher`

    References:
        .. [2] Pedregosa, Gidel, Adaptive Three Operator Splitting in
               Proceedings of the 35th International Conference on Machine
               Learning, Proceedings of Machine Learning Research., J. Dy, A.
               Krause, Eds. (PMLR, 2018), pp. 4085–4094.

    Examples:

        Usage is very similar to :meth:`mushi.optimization.AccProxGrad`, except
        that two non-smooth terms (and their associated proximal operators) may
        be specified.

        >>> import mushi.optimization as opt
        >>> import numpy as np

        We'll use a squared loss term, a Lasso term and a box constraint for
        this example.

        Define :math:`g(x)` and :math:`\boldsymbol\nabla g(x)`:

        >>> def g(x):
        ...     return 0.5 * np.sum(x ** 2)

        >>> def grad(x):
        ...     return x

        Define :math:`h_1(x)` and :math:`\mathrm{prox}_{h_1}(u)`.
        We will use a Lasso term and the corresponding soft thresholding
        operator:

        >>> def h1(x):
        ...     return np.linalg.norm(x, 1)

        >>> def prox1(u, s):
        ...     return np.sign(u) * np.clip(np.abs(u) - s, 0, None)

        Define :math:`h_2(x)` and :math:`\mathrm{prox}_{h_2}(u)`. We use a simple
        box constraint on one dimension, although note that this is quite
        artificial, since such constraints don't require operator splitting.

        >>> def h2(x):
        ...     if x[0] < 1:
        ...         return np.inf
        ...     return 0

        >>> def prox2(u, s):
        ...     return np.clip(u, np.array([1, -np.inf]), None)

        Initialize optimizer and define initial point

        >>> threeop = opt.ThreeOpProxGrad(g, grad, h1, prox1, h2, prox2)
        >>> x = np.zeros(2)

        Run optimization

        >>> threeop.run(x)
        array([1., 0.])

        Evaluate cost at the solution point

        >>> threeop.f()
        1.5

    """

    def __init__(self,
                 g: Callable[[np.ndarray], np.float64],
                 grad: Callable[[np.ndarray], np.float64],
                 h1: Callable[[np.ndarray], np.float64],
                 prox1: Callable[[np.ndarray, np.float64], np.float64],
                 h2: Callable[[np.ndarray], np.float64],
                 prox2: Callable[[np.ndarray, np.float64], np.float64],
                 verbose: bool = False,
                 **line_search_kwargs):
        super().__init__(g, grad, h1, prox1, verbose=verbose,
                         **line_search_kwargs)
        self.h2 = h2
        self.prox2 = prox2

    def f(self):
        return super().f() + self.h2(self.x)

    def _initialize(self, x: np.ndarray) -> None:
        super()._initialize(x)
        # dual variable
        self.u = np.zeros_like(self.q)

    def _step(self) -> None:
        """step with backtracking line search"""
        self._check_x()
        # evaluate differentiable part of objective
        g1 = self.g(self.q)
        grad1 = self.grad(self.q)
        if not np.all(np.isfinite(grad1)):
            raise RuntimeError(f'invalid gradient:\n{grad1}')
        # Armijo line search
        for line_iter in range(self.max_line_iter):
            # new point via prox-gradient of momentum point
            self.x = self.prox(self.q - self.s * (self.u + grad1), self.s)
            # quadratic approximation of objective
            Q = (g1 + (grad1 * (self.x - self.q)).sum()
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

    Example:

        >>> import mushi.optimization as opt
        >>> import numpy as np

        Define piecewise constant input signal

        >>> x = np.zeros(10)
        >>> x[3:7] = 1
        >>> x
        array([0., 0., 0., 1., 1., 1., 1., 0., 0., 0.])

        Corrupt the signal with additive noise

        >>> np.random.seed(0)
        >>> x += .1 * np.random.randn(10)

        Initialize and run 0-th order trend filter on input signal

        >>> tf = opt.TrendFilter((0,), (1e-1,))

        >>> tf.run(x)
        array([0.13810345, 0.13809355, 0.13809775, 1.10542391, 1.1054212 ,
               0.99863353, 0.99864752, 0.03853261, 0.03853295, 0.03853668])

    """

    def __init__(self, ks: Tuple[int], lambdas: Tuple[np.float64],
                 rhos: Tuple[np.float64] = None, verbose: bool = False):
        self.k = tuple(ks)
        self.λ = tuple(lambdas)
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
        self._check_x()
        fit = 0.5 * np.sum((self.x - self.y) ** 2)
        penalty = sum(λ * np.linalg.norm(np.diff(self.x, k + 1, axis=0), 1)
                      for k, λ in zip(self.k, self.λ))
        return fit + penalty

    def _initialize(self, x: np.ndarray) -> None:
        # input signal
        self.y = x
        if self.y.ndim == 1:
            self.y = self.y[:, np.newaxis]
        self.n = self.y.shape[0]
        # initialize solution point (β in the paper cited)
        self.x = np.zeros_like(self.y)
        # initialize Lagrangian dual variables
        self.α = [np.zeros_like(self.y)[:(self.n - k)] for k in self.k]
        self.u = [np.zeros_like(self.y)[:(self.n - k)] for k in self.k]

        self.D, self.DTD = self._D_DTD(self.n, self.k)
        self.c = self._choleskify(self.n, self.k, self.ρ)

    def _step(self):
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
