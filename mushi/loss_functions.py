r"""Loss functions functions

"""

import numpy as onp
import jax.numpy as np
from jax.ops import index, index_update
from typing import List


def prf(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""Poisson random field loss

    Args:
        E: expectation :math:`E[X]`
        X: data

    """
    return (E - X * np.log(E)).sum()


def dkl(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""generalized Kullback-Liebler divergence, a Bregman divergence (ignores
    constant term)

    Args:
        E: expectation :math:`E[X]`
        X: data

    """
    return (X * np.log(X / E) - X + E).sum()


def lsq(E: np.ndarray, X: np.ndarray) -> np.float64:
    r"""Least-squares loss

    Args:
        E: expectation :math:`E[X]`
        X: data

    """
    return (1 / 2) * ((E - X) ** 2).sum()


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
    result = index_update(result, index[:-1],
                          np.squeeze(A2[np.newaxis, :]
                                     @ np.exp(np.cumsum(logu[np.newaxis, :-1],
                                                        axis=1)) ** binom_vec))

    assert np.all(np.isfinite(result))

    return result


def revcomp(mutation_type: str) -> str:
    """reverse complement mutation type

    Args:
        mutation_type: mutation type string, e.g. AAA>ACA
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    anc, der = mutation_type.split('>')
    anc = ''.join(complement[b] for b in reversed(anc))
    der = ''.join(complement[b] for b in reversed(der))
    return anc + '>' + der


def misid_partners(mutation_types: List[str]) -> List[int]:
    """ancestral state misidentification indices

    Args:
        mutation_types: list of mutation type strings, e.g. [AAA>ACA, ...]

    """
    to_pair = list(enumerate(mutation_types))
    pair_idxs = [-1] * len(to_pair)
    while to_pair:
        i, mutype1 = to_pair[0]
        match = False
        match_revcomp = False
        for j, mutype2 in to_pair[1:]:
            if mutype1.split('>') == mutype2.split('>')[::-1]:
                match = True
                j_match = j
            elif mutype1.split('>') == mutype2.split('>')[::-1]:
                match_revcomp = True
                j_match_revcomp = j
        if match:
            j = j_match
        elif match_revcomp:
            j = j_match_revcomp
        else:
            raise ValueError('no ancestral misidentification partner found '
                             f'for mutation type {mutype1}')
        pair_idxs[i] = j
        pair_idxs[j] = i
        del to_pair[i]
        del to_pair[j]
        assert set(pair_idxs) == set(range(len(mutation_types)))
        return pair_idxs

def mutype_misid(mutation_types: List[str]):
    """mutation type misidentification operator

    Args:
        mutation_types: list of mutation type strings, e.g. [AAA>ACA, ...]
    """
    AM_mut = np.zeros((len(mutation_types), len(mutation_types)))
    if isinstance(mutation_types[0], int):
        # assume consecutive pairs are misid partners if types are integers
        for j in range(0, len(mutation_types), 2):
            AM_mut = index_update(AM_mut, index[j + 1, j], 1)
            AM_mut = index_update(AM_mut, index[j, j + 1], 1)
    else:
        for j, i in enumerate(misid_partners(mutation_types)):
            AM_mut = index_update(AM_mut, index[i, j], 1)
    return AM_mut

def fold(x: np.ndarray) -> np.ndarray:
    """transform SFS to folded SFS"""
    n = len(x) + 1
    x = (x + x[::-1])[:(n // 2)]
    if n % 2 == 0:
        x = index_update(x, index[-1], x[-1] // 2)
    return x
