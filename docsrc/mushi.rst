.. currentmodule:: mushi

Public API: ``mushi`` package
=============================

Classes
-------

Primary classes for working with :math:`k`-SFS data to infer demography :math:`\eta(t)\equiv 2N(t)` and mutation spectrum history :math:`\boldsymbol\mu(t)`.

.. autosummary::
    :toctree: stubs

    kSFS
    eta
    mu


Submodules
-----------

Submodules that the main classes rely on, and may be of interest to some users

.. autosummary::
    :toctree: stubs

    mushi.optimization
    mushi.composition
    mushi.utils





.. autoclass:: kSFS
    :members:

.. autoclass:: eta
    :members:

.. autoclass:: mu
    :members:


.. automodule:: mushi.optimization
    :members:

.. automodule:: mushi.utils
    :members:

.. automodule:: mushi.composition
    mushi.composition.clr
    mushi.composition.clr_inv
    mushi.composition.ilr
    mushi.composition.ilr_inv
    mushi.composition.centralize
