r"""Utility functions

"""

import numpy as onp
import jax.numpy as np
from typing import List


def C(n: int) -> np.ndarray:
    r"""The combinatorial matrix :math:`\mathbf C` defined in the paper's
    appendix.

    Args:
        n: the number of sampled haplotypes :math:`n`

    Returns:
        :math:`(n-1)\times(n-1)` matrix

    """
    W1 = np.zeros((n - 1, n - 1))
    W2 = np.zeros((n - 1, n - 1))
    b = np.arange(1, n - 1 + 1)
    # j = 2
    W1 = W1.at[:, 0].set(6 / (n + 1))
    W2 = W2.at[:, 0].set(0)
    # j = 3
    W1 = W1.at[:, 1].set(10 * (5 * n - 6 * b - 4) / (n + 2) / (n + 1))
    W2 = W2.at[:, 1].set((20 * (n - 2)) / (n+1) / (n+2))
    for col in range(n - 3):
        # this cast is crucial for floating point precision
        j = np.float64(col + 2)
        # procedurally generated by Zeilberger's algorithm in Mathematica
        W1 = W1.at[:, col + 2].set(-((-((-1 + j)*(1 + j)**2*(3 + 2*j)*(j - n)*(4 + 2*j - 2*b*j + j**2 - b*j**2 + 4*n + 2*j*n + j**2*n)*W1[:, col]) - (-1 + 2*j)*(3 + 2*j)*(-4*j - 12*b*j - 4*b**2*j - 6*j**2 - 12*b*j**2 - 2*b**2*j**2 - 4*j**3 + 4*b**2*j**3 - 2*j**4 + 2*b**2*j**4 + 4*n + 2*j*n - 6*b*j*n + j**2*n - 9*b*j**2*n - 2*j**3*n - 6*b*j**3*n - j**4*n - 3*b*j**4*n + 4*n**2 + 6*j*n**2 + 7*j**2*n**2 + 2*j**3*n**2 + j**4*n**2)*W1[:, col + 1])/(j**2*(2 + j)*(-1 + 2*j)*(1 + j + n)*(3 + b + j**2 - b*j**2 + 3*n + j**2*n))))  # noqa: E501
        W2 = W2.at[:, col + 2].set(((-1 + j)*(1 + j)*(2 + j)*(3 + 2*j)*(j - n)*(1 + j - n)*(1 + j + n)*W2[:, col] + (-1 + 2*j)*(3 + 2*j)*(1 + j - n)*(j + n)*(2 - j - 2*b*j - j**2 - 2*b*j**2 + 2*n + j*n + j**2*n)*W2[:, col + 1])/((-1 + j)*j*(2 + j)*(-1 + 2*j)*(j - n)*(j + n)*(1 + j + n)))   # noqa: E501

    return W1 - W2


def M(n: int, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""The matrix :math:`\mathbf M` defined in the paper's appendix

    Args:
        n: the number of sampled haplotypes :math:`n`
        t: time grid, starting at zero and ending at np.inf
        y: population size in each epoch

    Returns:
        :math:`(n-1)\times m` matrix, where :math:`m` is the number of epochs
        (the length of the ``y`` argument)

    """
    # epoch durations
    s = np.diff(t)
    # we handle the final infinite epoch carefully to facilitate autograd
    u = np.exp(-s[:-1] / y[:-1])
    u = np.concatenate((np.array([1]), u, np.array([0])))

    n_range = np.arange(2, n + 1)
    binom_vec = n_range * (n_range - 1) / 2

    return np.exp(binom_vec[:, np.newaxis]
                  * np.cumsum(np.log(u[np.newaxis, :-1]), axis=1)
                  - np.log(binom_vec[:, np.newaxis])) \
        @ (np.eye(len(y), k=0) - np.eye(len(y), k=-1)) \
        @ np.diag(y)


def tmrca_sf(t: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """The survival function of the TMRCA at each time point

    Args:
        t: time grid (including zero and infinity)
        y: effective population size in each epoch
        n: number of sampled haplotypes

    """
    # epoch durations
    s = np.diff(t)
    logu = -s / y
    logu = np.concatenate((np.array([0]), logu))
    # the A_2j are the product of this matrix
    # NOTE: using letter  "l" as a variable name to match text
    l = onp.arange(2, n + 1)[:, onp.newaxis]  # noqa: E741
    with onp.errstate(divide='ignore'):
        A2_terms = l * (l - 1) / (l * (l - 1) - l.T * (l.T - 1))
    onp.fill_diagonal(A2_terms, 1)
    A2 = np.prod(A2_terms, axis=0)

    binom_vec = l * (l - 1) / 2

    result = np.zeros(len(t))
    result = result.at[:-1].set(np.squeeze(A2[np.newaxis, :]
                                @ np.exp(np.cumsum(logu[np.newaxis, :-1],
                                                   axis=1)) ** binom_vec))

    assert np.all(np.isfinite(result))

    return result


complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def revcomp(seq: str) -> str:
    """Reverse complement of nucleotide sequence

    Args:
        seq: nucleotide string (all caps ACGT)
    """
    return ''.join(complement[b] for b in reversed(seq))


def misid_partners(mutation_types: List[str]) -> List[int]:
    """Ancestral state misidentification partner indices. Mutation type ``None``
    is self-partnered.

    Args:
        mutation_types: list of mutation type strings, e.g. [AAA>ACA, ...]

    Examples:

        >>> from mushi import utils

        >>> utils.misid_partners(['TCC>TTC', 'GAA>GGA', None])
        [1, 0, 2]

    """
    to_pair = list(enumerate(mutation_types))
    pair_idxs = [-1] * len(to_pair)
    while to_pair:
        i, mutype1 = to_pair[0]
        if mutype1 is None:
            pair_idxs[i] = i
            del to_pair[0]
        else:
            anc1, der1 = mutype1.split('>')
            match = False
            for new_idx, (j, mutype2) in enumerate(to_pair):
                anc2, der2 = mutype2.split('>')
                if (anc1, der1) in ((der2, anc2), (revcomp(der2), revcomp(anc2))):
                    match = True
                    break
            if not match:
                raise ValueError('no ancestral misidentification partner found '
                                 f'for mutation type {mutype1}')
            pair_idxs[i] = j
            pair_idxs[j] = i
            del to_pair[new_idx]
            if new_idx != 0:
                del to_pair[0]
    assert set(pair_idxs) == set(range(len(mutation_types)))
    return pair_idxs


def mutype_misid(mutation_types: List[str]) -> np.ndarray:
    """mutation type misidentification operator

    Args:
        mutation_types: list of mutation type strings, e.g. [AAA>ACA, ...]

    Examples:

        >>> from mushi import utils

        >>> utils.mutype_misid(['TCC>TTC', 'GAA>GGA', None])
        DeviceArray([[0., 1., 0.],
                     [1., 0., 0.],
                     [0., 0., 1.]], dtype=float64)

    """
    AM_mut = np.zeros((len(mutation_types), len(mutation_types)))
    for j, i in enumerate(misid_partners(mutation_types)):
        AM_mut = AM_mut.at[i, j].set(1)
    return AM_mut


def fold(x: np.ndarray) -> np.ndarray:
    """transform SFS to folded SFS

    Args:
        func: loss function name from loss_functions module

    Examples:

        >>> from mushi import utils
        >>> import jax.numpy as np

        >>> sfs = np.array([1000, 100, 10])

        >>> utils.fold(sfs)
        DeviceArray([1010,  100], dtype=int64)

    """
    n = len(x) + 1
    x = (x + x[::-1])[:(n // 2)]
    if n % 2 == 0:
        x = x.at[-1].set(x[-1] // 2)
    return x
