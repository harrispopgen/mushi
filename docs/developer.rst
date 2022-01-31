Open source code repository
===========================

All code is freely available at `<https://github.com/harrispopgen/mushi>`_

Developer tools
===============

Developer install::

  make install

Run tests::

  make test

Format code::

  make format

Lint::

  make lint

Build docs locally (you can then see the generated documentation in ``docs/_build/html/index.html``)::

  make docs

Docs are automatically deployed to github pages via a workflow on push to the master branch.


Test notebook
=============

A test notebook that simulates under the linkage-free forward model (where the PRF composite likelihood is not approximate): `<notebooks/simulation.ipynb>`_

Note that this notebook is used to build a docs page.

Todo list
=========

.. todolist::
