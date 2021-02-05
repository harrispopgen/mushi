Open source code repository
===========================

All code is freely available at `<https://github.com/harrispopgen/mushi>`_


Update documentation
====================

Inspired by: https://www.docslikecode.com/articles/github-pages-python-sphinx/

Go to the ``docsrc`` directory::

  cd docsrc

Environment
-----------

Create and activate the ``mushidocs`` conda environment::

  conda env create -f env.yml
  conda activate mushidocs

Install ``mushi`` itself from the local copy in the parent directory::

  pip install -e ..

Modify notebooks in the ``notebooks`` directory as needed.

.. note::

  Executing builds (below) after modifying notebooks can take a very long time
  if compute-heavy notebooks need to be recompiled.

Run doctests
------------

From the ``docsrc`` dir::

  make doctest

Local build
-----------

From the ``docsrc`` dir::

  make html

You can then see the generated documentation in ``docsrc/_build/index.html``.

Github Pages build
------------------

From the ``docsrc`` dir::

  make github

You can then see the generated documentation in
``docs/index.html``.

Test notebook
=============

A test notebook that simulates under the linkage-free forward model (where the PRF composite likelihood is not approximate): `<notebooks/simulation.ipynb>`_

Note that this notebook is used to build a docs page.

Todo list
=========

.. todolist::
