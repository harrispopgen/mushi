Mutation Spectrum History Inference
###################################

``mushi`` is a Python package for nonparametric coalescent inference of demographic
history and mutation spectrum history (MuSH) from genomic variation data.
Both demographic history and MuSH are encoded in sample allele frequency
spectra as joint inverse problems, and ``mushi`` uses fast optimization methods
to infer these histories from data without model specification.

.. toctree::
  :maxdepth: 1
  :caption: User Guide

  install
  notebooks/quickstart
  cite

.. toctree::
  :maxdepth: 1
  :caption: Notebooks

  notebooks/simulation
  notebooks/1KG
  notebooks/observability

.. todo:: set up Google collab for notebooks

.. toctree::
   :maxdepth: 1
   :caption: Notes

   CHANGELOG
   faq

.. toctree::
   :maxdepth: 3
   :caption: API Documentation

   mushi

.. toctree::
  :maxdepth: 1
  :caption: Developer Documentation

  developer

.. todo:: document CLI

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
