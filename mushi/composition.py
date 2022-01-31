r"""
Functions for compositional data analysis.

It was copied from the :mod:`skbio.stats.composition` module and edited slightly to
enable compatibility with the JAX package.

Many 'omics datasets are inherently compositional - meaning that they
are best interpreted as proportions or percentages rather than
absolute counts.

Formally, :math:`x` is a composition if :math:`\sum_{i=0}^D x_{i} = c`
and :math:`x_{i} > 0`, :math:`1 \leq i \leq D` and :math:`c` is a real
valued constant and there are :math:`D` components for each
composition. In this module :math:`c=1`. Compositional data can be
analyzed using Aitchison geometry. [1]_

However, in this framework, standard real Euclidean operations such as
addition and multiplication no longer apply. Only operations such as
perturbation and power can be used to manipulate this data.

This module allows two styles of manipulation of compositional data.
Compositional data can be analyzed using perturbation and power
operations, which can be useful for simulation studies. The
alternative strategy is to transform compositional data into the real
space.  Right now, the centre log ratio transform (clr) and
the isometric log ratio transform (ilr) [2]_ can be used to accomplish
this. This transform can be useful for performing standard statistical
tools such as parametric hypothesis testing, regressions and more.

The major caveat of using this framework is dealing with zeros.  In
the Aitchison geometry, only compositions with nonzero components can
be considered. The multiplicative replacement technique [3]_ can be
used to substitute these zeros with small pseudocounts without
introducing major distortions to the data.

References
----------
.. [1] V. Pawlowsky-Glahn, "Lecture Notes on Compositional Data Analysis"
   (2007)

.. [2] J. J. Egozcue.,  "Isometric Logratio Transformations for
   Compositional Data Analysis" Mathematical Geology, 35.3 (2003)

.. [3] J. A. Martin-Fernandez,  "Dealing With Zeros and Missing Values in
   Compositional Data Sets Using Nonparametric Imputation",
   Mathematical Geology, 35.3 (2003)


Examples
--------

>>> import numpy as np
>>> import mushi.composition as cmp

Consider a very simple environment with only 3 species. The species
in the environment are equally distributed and their proportions are
equivalent:

>>> otus = np.array([1./3, 1./3., 1./3])

Suppose that an antibiotic kills off half of the population for the
first two species, but doesn't harm the third species. Then the
perturbation vector would be as follows

>>> antibiotic = np.array([1./2, 1./2, 1])

And the resulting perturbation would be

>>> cmp.perturb(otus, antibiotic)
DeviceArray([0.25, 0.25, 0.5 ], dtype=float64)

"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function
import jax.numpy as np
import numpy as onp
import pandas as pd
import scipy.stats


def closure(mat):
    """Performs closure to ensure that all elements add up to 1.

    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components

    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> cmp.closure(X)
    DeviceArray([[0.2, 0.2, 0.6],
                 [0.4, 0.4, 0.2]], dtype=float64)
    """
    mat = np.atleast_2d(mat)
    # if np.any(mat < 0):
    #     raise ValueError("Cannot have negative proportions")
    # if mat.ndim > 2:
    #     raise ValueError("Input matrix can only have two dimensions or less")
    # if np.all(mat == 0, axis=1).sum() > 0:
    #     raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()


def multiplicative_replacement(mat, delta=None):
    r"""Replace all zeros with small non-zero values

    It uses the multiplicative replacement strategy [1]_ ,
    replacing zeros with a small positive :math:`\delta`
    and ensuring that the compositions still add up to 1.


    Parameters
    ----------
    mat: array_like
       a matrix of proportions where
       rows = compositions and
       columns = components
    delta: float, optional
       a small number to be used to replace zeros
       If delta is not specified, then the default delta is
       :math:`\delta = \frac{1}{N^2}` where :math:`N`
       is the number of components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Raises
    ------
    ValueError
       Raises an error if negative proportions are created due to a large
       `delta`.

    Notes
    -----
    This method will result in negative proportions if a large delta is chosen.

    References
    ----------
    .. [1] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
           Compositional Data Sets Using Nonparametric Imputation"


    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> X = np.array([[.2,.4,.4, 0],[0,.5,.5,0]])
    >>> cmp.multiplicative_replacement(X)
    DeviceArray([[0.1875, 0.375 , 0.375 , 0.0625],
                 [0.0625, 0.4375, 0.4375, 0.0625]], dtype=float64)

    """
    mat = closure(mat)
    z_mat = mat == 0

    num_feats = mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1.0 / num_feats) ** 2

    zcnts = 1 - tot * delta
    if np.any(zcnts) < 0:
        raise ValueError(
            "The multiplicative replacment created negative "
            "proportions. Consider using a smaller `delta`."
        )
    mat = np.where(z_mat, delta, zcnts * mat)
    return mat.squeeze()


def perturb(x, y):
    r"""
    Performs the perturbation operation.

    This operation is defined as

    .. math::
        x \oplus y = C[x_1 y_1, \ldots, x_D y_D]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> cmp.perturb(x,y)
    DeviceArray([0.0625, 0.1875, 0.5   , 0.25  ], dtype=float64)

    """
    x, y = closure(x), closure(y)
    return closure(x * y)


def perturb_inv(x, y):
    r"""
    Performs the inverse perturbation operation.

    This operation is defined as

    .. math::
        x \ominus y = C[x_1 y_1^{-1}, \ldots, x_D y_D^{-1}]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]


    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> x = np.array([.1,.3,.4, .2])
    >>> y = np.array([1./6,1./6,1./3,1./3])
    >>> cmp.perturb_inv(x,y)
    DeviceArray([0.14285714, 0.42857143, 0.28571429, 0.14285714], dtype=float64)
    """
    x, y = closure(x), closure(y)
    return closure(x / y)


def power(x, a):
    r"""
    Performs the power operation.

    This operation is defined as follows

    .. math::
        `x \odot a = C[x_1^a, \ldots, x_D^a]

    :math:`C[x]` is the closure operation defined as

    .. math::
        C[x] = \left[\frac{x_1}{\sum_{i=1}^{D} x_i},\ldots,
                     \frac{x_D}{\sum_{i=1}^{D} x_i} \right]

    for some :math:`D` dimensional real vector :math:`x` and
    :math:`D` is the number of components for every composition.

    Parameters
    ----------
    x : array_like, float
        a matrix of proportions where
        rows = compositions and
        columns = components
    a : float
        a scalar float

    Returns
    -------
    numpy.ndarray, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> x = np.array([.1,.3,.4, .2])
    >>> cmp.power(x, .1)
    DeviceArray([0.23059566, 0.25737316, 0.26488486, 0.24714631], dtype=float64)

    """
    x = closure(x)
    return closure(x**a).squeeze()


def inner(x, y):
    r"""
    Calculates the Aitchson inner product.

    This inner product is defined as follows

    .. math::
        \langle x, y \rangle_a =
        \frac{1}{2D} \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{D}
        \ln\left(\frac{x_i}{x_j}\right) \ln\left(\frac{y_i}{y_j}\right)

    Parameters
    ----------
    x : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components
    y : array_like
        a matrix of proportions where
        rows = compositions and
        columns = components

    Returns
    -------
    numpy.ndarray
         inner product result

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> x = np.array([.1, .3, .4, .2])
    >>> y = np.array([.2, .4, .2, .2])
    >>> cmp.inner(x, y)  # doctest: +ELLIPSIS
    DeviceArray(0.21078525, dtype=float64)
    """
    x = closure(x)
    y = closure(y)
    a, b = clr(x), clr(y)
    return a.dot(b.T)


def clr(mat):
    r"""
    Performs centre log ratio transformation.

    This function transforms compositions from Aitchison geometry to
    the real space. The :math:`clr` transform is both an isometry and an
    isomorphism defined on the following spaces

    :math:`clr: S^D \rightarrow U`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    It is defined for a composition :math:`x` as follows:

    .. math::
        clr(x) = \ln\left[\frac{x_1}{g_m(x)}, \ldots, \frac{x_D}{g_m(x)}\right]

    where :math:`g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D}` is the geometric
    mean of :math:`x`.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> x = np.array([.1, .3, .4, .2])
    >>> cmp.clr(x)
    DeviceArray([-0.79451346,  0.30409883,  0.5917809 , -0.10136628], dtype=float64)

    """
    mat = closure(mat)
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()


def clr_inv(mat):
    r"""
    Performs inverse centre log ratio transformation.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`clr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`clr^{-1}: U \rightarrow S^D`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    This transformation is defined as follows

    .. math::
        clr^{-1}(x) = C[\exp( x_1, \ldots, x_D)]

    Parameters
    ----------
    mat : array_like, float
       a matrix of real values where
       rows = transformed compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         inverse clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> x = np.array([.1, .3, .4, .2])
    >>> cmp.clr_inv(x)
    DeviceArray([0.21383822, 0.26118259, 0.28865141, 0.23632778], dtype=float64)

    """
    return closure(np.exp(mat))


def ilr(mat, basis=None, check=True):
    r"""
    Performs isometric log ratio transformation.

    This function transforms compositions from Aitchison simplex to
    the real space. The :math: ilr` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr: S^D \rightarrow \mathbb{R}^{D-1}`

    The ilr transformation is defined as follows

    .. math::
        ilr(x) =
        [\langle x, e_1 \rangle_a, \ldots, \langle x, e_{D-1} \rangle_a]

    where :math:`[e_1,\ldots,e_{D-1}]` is an orthonormal basis in the simplex.

    If an orthornormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.

    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components

    basis: numpy.ndarray, float, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> x = np.array([.1, .3, .4, .2])
    >>> cmp.ilr(x)
    DeviceArray([-0.7768362 , -0.68339802,  0.11704769], dtype=float64)

    """
    mat = closure(mat)
    if basis is None:
        basis = _gram_schmidt_basis(mat.shape[-1])
    elif check:
        _check_orthogonality(clr_inv(basis))
    return inner(mat, clr_inv(basis))


def ilr_inv(mat, basis):
    r"""
    Performs inverse isometric log ratio transform.

    This function transforms compositions from the real space to
    Aitchison geometry. The :math:`ilr^{-1}` transform is both an isometry,
    and an isomorphism defined on the following spaces

    :math:`ilr^{-1}: \mathbb{R}^{D-1} \rightarrow S^D`

    The inverse ilr transformation is defined as follows

    .. math::
        ilr^{-1}(x) = \bigoplus\limits_{i=1}^{D-1} x \odot e_i

    where :math:`[e_1,\ldots, e_{D-1}]` is an orthonormal basis in the simplex.

    If an orthornormal basis isn't specified, the J. J. Egozcue orthonormal
    basis derived from Gram-Schmidt orthogonalization will be used by
    default.


    Parameters
    ----------
    mat: numpy.ndarray, float
       a matrix of transformed proportions where
       rows = compositions and
       columns = components

    basis: numpy.ndarray, float, optional
        orthonormal basis for Aitchison simplex
        defaults to J.J.Egozcue orthonormal basis

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> x = np.array([.1, .3, .6,])
    >>> basis = cmp._gram_schmidt_basis(4)
    >>> cmp.ilr_inv(x, basis)
    DeviceArray([0.34180297, 0.29672718, 0.22054469, 0.14092516], dtype=float64)

    """

    return clr_inv(np.dot(mat, basis))


def centralize(mat):
    r"""Center data around its geometric average.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         centered composition matrix

    Examples
    --------
    >>> import numpy as np
    >>> import mushi.composition as cmp
    >>> X = np.array([[.1,.3,.4, .2],[.2,.2,.2,.4]])
    >>> cmp.centralize(X)
    DeviceArray([[0.17445763, 0.30216948, 0.34891526, 0.17445763],
                 [0.32495488, 0.18761279, 0.16247744, 0.32495488]],            dtype=float64)

    """
    mat = closure(mat)
    cen = scipy.stats.gmean(mat, axis=0)
    return perturb_inv(mat, cen)


def ancom(
    table,
    grouping,
    alpha=0.05,
    tau=0.02,
    theta=0.1,
    multiple_comparisons_correction=None,
    significance_test=None,
):
    r"""Performs a differential abundance test using ANCOM.

    This is done by calculating pairwise log ratios between all features
    and performing a significance test to determine if there is a significant
    difference in feature ratios with respect to the variable of interest.

    In an experiment with only two treatments, this test tests the following
    hypothesis for feature :math:`i`

    .. math::

        H_{0i}: \mathbb{E}[\ln(u_i^{(1)})] = \mathbb{E}[\ln(u_i^{(2)})]

    where :math:`u_i^{(1)}` is the mean abundance for feature :math:`i` in the
    first group and :math:`u_i^{(2)}` is the mean abundance for feature
    :math:`i` in the second group.

    Parameters
    ----------
    table : pd.DataFrame
        A 2D matrix of strictly positive values (i.e. counts or proportions)
        where the rows correspond to samples and the columns correspond to
        features.
    grouping : pd.Series
        Vector indicating the assignment of samples to groups.  For example,
        these could be strings or integers denoting which group a sample
        belongs to.  It must be the same length as the samples in `table`.
        The index must be the same on `table` and `grouping` but need not be
        in the same order.
    alpha : float, optional
        Significance level for each of the statistical tests.
        This can can be anywhere between 0 and 1 exclusive.
    tau : float, optional
        A constant used to determine an appropriate cutoff.
        A value close to zero indicates a conservative cutoff.
        This can can be anywhere between 0 and 1 exclusive.
    theta : float, optional
        Lower bound for the proportion for the W-statistic.
        If all W-statistics are lower than theta, then no features
        will be detected to be differentially significant.
        This can can be anywhere between 0 and 1 exclusive.
    multiple_comparisons_correction : {None, 'holm-bonferroni'}, optional
        The multiple comparison correction procedure to run.  If None,
        then no multiple comparison correction procedure will be run.
        If 'holm-boniferroni' is specified, then the Holm-Boniferroni
        procedure [1]_ will be run.
    significance_test : function, optional
        A statistical significance function to test for significance between
        classes.  This function must be able to accept at least two 1D
        array_like arguments of floats and returns a test statistic and a
        p-value. By default ``scipy.stats.f_oneway`` is used.

    Returns
    -------
    pd.DataFrame
        A table of features, their W-statistics and whether the null hypothesis
        is rejected.

        `"W"` is the W-statistic, or number of features that a single feature
        is tested to be significantly different against.

        `"reject"` indicates if feature is significantly different or not.

    See Also
    --------
    multiplicative_replacement
    scipy.stats.ttest_ind
    scipy.stats.f_oneway
    scipy.stats.wilcoxon
    scipy.stats.kruskal

    Notes
    -----
    The developers of this method recommend the following significance tests
    ([2]_, Supplementary File 1, top of page 11): the standard parametric
    t-test (``scipy.stats.ttest_ind``) or one-way ANOVA
    (``scipy.stats.f_oneway``) if the number of groups is greater
    than 2, or non-parametric variants such as Wilcoxon rank sum
    (``scipy.stats.wilcoxon``) or Kruskal-Wallis (``scipy.stats.kruskal``)
    if the number of groups is greater than 2.  Because one-way ANOVA is
    equivalent to the standard t-test when the number of groups is two,
    we default to ``scipy.stats.f_oneway`` here, which can be used when
    there are two or more groups.  Users should refer to the documentation
    of these tests in SciPy to understand the assumptions made by each test.

    This method cannot handle any zero counts as input, since the logarithm
    of zero cannot be computed.  While this is an unsolved problem, many
    studies have shown promising results by replacing the zeros with pseudo
    counts. This can be also be done via the ``multiplicative_replacement``
    method.

    References
    ----------
    .. [1] Holm, S. "A simple sequentially rejective multiple test procedure".
       Scandinavian Journal of Statistics (1979), 6.
    .. [2] Mandal et al. "Analysis of composition of microbiomes: a novel
       method for studying microbial composition", Microbial Ecology in Health
       & Disease, (2015), 26.

    Examples
    --------
    First import all of the necessary modules:

    >>> import mushi.composition as cmp
    >>> import pandas as pd

    Now let's load in a pd.DataFrame with 6 samples and 7 unknown bacteria:

    >>> table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
    ...                       [9,  11, 12, 10, 10, 10, 10],
    ...                       [1,  11, 10, 11, 10, 5,  9],
    ...                       [22, 21, 9,  10, 10, 10, 10],
    ...                       [20, 22, 10, 10, 13, 10, 10],
    ...                       [23, 21, 14, 10, 10, 10, 10]],
    ...                      index=['s1','s2','s3','s4','s5','s6'],
    ...                      columns=['b1','b2','b3','b4','b5','b6','b7'])

    Then create a grouping vector.  In this scenario, there
    are only two classes, and suppose these classes correspond to the
    treatment due to a drug and a control.  The first three samples
    are controls and the last three samples are treatments.

    >>> grouping = pd.Series([0, 0, 0, 1, 1, 1],
    ...                      index=['s1','s2','s3','s4','s5','s6'])

    Now run ``ancom`` and see if there are any features that have any
    significant differences between the treatment and the control.

    >>> results = cmp.ancom(table, grouping) # doctest: +SKIP
    >>> results['W'] # doctest: +SKIP
    b1    0
    b2    4
    b3    1
    b4    1
    b5    1
    b6    0
    b7    1
    Name: W, dtype: np.int64

    The W-statistic is the number of features that a single feature is tested
    to be significantly different against.  In this scenario, `b2` was detected
    to have significantly different abundances compared to four of the other
    species. To summarize the results from the W-statistic, let's take a look
    at the results from the hypothesis test:

    >>> results['reject'] # doctest: +SKIP
    b1    False
    b2     True
    b3    False
    b4    False
    b5    False
    b6    False
    b7    False
    Name: reject, dtype: bool

    From this we can conclude that only `b2` was significantly
    different between the treatment and the control.

    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "`table` must be a `pd.DataFrame`, " "not %r." % type(table).__name__
        )
    if not isinstance(grouping, pd.Series):
        raise TypeError(
            "`grouping` must be a `pd.Series`," " not %r." % type(grouping).__name__
        )

    if np.any(table <= 0):
        raise ValueError(
            "Cannot handle zeros or negative values in `table`. "
            "Use pseudo counts or ``multiplicative_replacement``."
        )

    if not 0 < alpha < 1:
        raise ValueError("`alpha`=%f is not within 0 and 1." % alpha)

    if not 0 < tau < 1:
        raise ValueError("`tau`=%f is not within 0 and 1." % tau)

    if not 0 < theta < 1:
        raise ValueError("`theta`=%f is not within 0 and 1." % theta)

    if multiple_comparisons_correction is not None:
        if multiple_comparisons_correction != "holm-bonferroni":
            raise ValueError(
                "%r is not an available option for "
                "`multiple_comparisons_correction`." % multiple_comparisons_correction
            )

    if (grouping.isnull()).any():
        raise ValueError("Cannot handle missing values in `grouping`.")

    if (table.isnull()).any().any():
        raise ValueError("Cannot handle missing values in `table`.")

    groups, _grouping = onp.unique(grouping, return_inverse=True)
    grouping = pd.Series(_grouping, index=grouping.index)
    num_groups = len(groups)

    if num_groups == len(grouping):
        raise ValueError(
            "All values in `grouping` are unique. This method cannot "
            "operate on a grouping vector with only unique values (e.g., "
            "there are no 'within' variance because each group of samples "
            "contains only a single sample)."
        )

    if num_groups == 1:
        raise ValueError(
            "All values the `grouping` are the same. This method cannot "
            "operate on a grouping vector with only a single group of samples"
            "(e.g., there are no 'between' variance because there is only a "
            "single group)."
        )

    if significance_test is None:
        significance_test = scipy.stats.f_oneway

    table_index_len = len(table.index)
    grouping_index_len = len(grouping.index)
    mat, cats = table.align(grouping, axis=0, join="inner")
    if len(mat) != table_index_len or len(cats) != grouping_index_len:
        raise ValueError("`table` index and `grouping` " "index must be consistent.")

    n_feat = mat.shape[1]

    _logratio_mat = _log_compare(mat.values, cats.values, significance_test)
    logratio_mat = _logratio_mat + _logratio_mat.T

    # Multiple comparisons
    if multiple_comparisons_correction == "holm-bonferroni":
        logratio_mat = np.apply_along_axis(_holm_bonferroni, 1, logratio_mat)
    np.fill_diagonal(logratio_mat, 1)
    W = (logratio_mat < alpha).sum(axis=1)
    c_start = W.max() / n_feat
    if c_start < theta:
        reject = np.zeros_like(W, dtype=bool)
    else:
        # Select appropriate cutoff
        cutoff = c_start - np.linspace(0.05, 0.25, 5)
        prop_cut = np.array([(W > n_feat * cut).mean() for cut in cutoff])
        dels = np.abs(prop_cut - np.roll(prop_cut, -1))
        dels[-1] = 0

        if (dels[0] < tau) and (dels[1] < tau) and (dels[2] < tau):
            nu = cutoff[1]
        elif (dels[0] >= tau) and (dels[1] < tau) and (dels[2] < tau):
            nu = cutoff[2]
        elif (dels[1] >= tau) and (dels[2] < tau) and (dels[3] < tau):
            nu = cutoff[3]
        else:
            nu = cutoff[4]
        reject = W >= nu * n_feat
    labs = mat.columns
    return pd.DataFrame(
        {"W": pd.Series(W, index=labs), "reject": pd.Series(reject, index=labs)}
    )


def _holm_bonferroni(p):
    """Performs Holm-Bonferroni correction for pvalues to account for multiple
    comparisons.

    Parameters
    ---------
    p: numpy.array
        array of pvalues

    Returns
    -------
    numpy.array
        corrected pvalues
    """
    K = len(p)
    sort_index = -np.ones(K, dtype=np.int64)
    sorted_p = np.sort(p)
    sorted_p_adj = sorted_p * (K - np.arange(K))
    for j in range(K):
        idx = (p == sorted_p[j]) & (sort_index < 0)
        num_ties = len(sort_index[idx])
        sort_index[idx] = np.arange(j, (j + num_ties), dtype=np.int64)

    sorted_holm_p = [min([max(sorted_p_adj[:k]), 1]) for k in range(1, K + 1)]
    holm_p = [sorted_holm_p[sort_index[k]] for k in range(K)]
    return holm_p


def _log_compare(mat, cats, significance_test=scipy.stats.ttest_ind):
    """Calculates pairwise log ratios between all features and performs a
    significiance test (i.e. t-test) to determine if there is a significant
    difference in feature ratios with respect to the variable of interest.

    Parameters
    ----------
    mat: np.array
       rows correspond to samples and columns correspond to
       features (i.e. OTUs)
    cats: np.array, float
       Vector of categories
    significance_test: function
        statistical test to run

    Returns:
    --------
    log_ratio : np.array
        log ratio pvalue matrix
    """
    r, c = mat.shape
    log_ratio = np.zeros((c, c))
    log_mat = np.log(mat)
    cs = np.unique(cats)

    def func(x):
        return significance_test(*[x[cats == k] for k in cs])

    for i in range(c - 1):
        ratio = (log_mat[:, i].T - log_mat[:, i + 1 :].T).T
        m, p = np.apply_along_axis(func, axis=0, arr=ratio)
        log_ratio[i, i + 1 :] = np.squeeze(np.array(p.T))
    return log_ratio


def _gram_schmidt_basis(n):
    """Builds clr transformed basis derived from gram schmidt
    orthogonalization.

    Parameters
    ----------
    n : int
        Dimension of the Aitchison simplex
    """
    basis = onp.zeros((n, n - 1))
    for j in range(n - 1):
        i = j + 1
        e = onp.array([(1 / i)] * i + [-1] + [0] * (n - i - 1)) * np.sqrt(i / (i + 1))
        basis[:, j] = e
    return basis.T


def _check_orthogonality(basis):
    """Checks to see if basis is truly orthonormal in the Aitchison simplex.

    Parameters
    ----------
    basis: numpy.ndarray
        basis in the Aitchison simplex
    """
    if not np.allclose(
        inner(basis, basis), np.identity(len(basis)), rtol=1e-4, atol=1e-6
    ):
        raise ValueError("Aitchison basis is not orthonormal")
