DemEnt
====
Demographic inference as an inverse problem

## Dependencies
  - python3.6+
  - numpy
  - scipy
  - matplotlib
  - jupyter

Code
---
- [`dement.py`](dement.py): Class implementing the coalescent model of the expected sample frequency spectrum, as described in Rosen et al., and adding the Poisson random field for a full generative model.
- [`regularization.ipynb`](regularization.ipynb): Jupyter notebook exploring penalized likelihood optimization.


References
----

The coalescent model we're using for the site-frequency spectrum:

 - Rosen, Zvi, Anand Bhaskar, Sebastien Roch, and Yun S. Song. 2018. [Geometry of the Sample Frequency Spectrum and the Perils of Demographic Inference.](http://dx.doi.org/10.1534/genetics.118.300733) _Genetics_, July, genetics.300733.2018.

 - Polanski, A., and M. Kimmel. 2003. [New Explicit Expressions for Relative Frequencies of Single-Nucleotide Polymorphisms with Application to Statistical Inference on Population Growth.](http://www.genetics.org/cgi/pmidlookup?view=long&pmid=14504247) _Genetics_ 165 (1): 427–36.

This paper shows that SFS-based methods and LD-based methods (sequential Markov coalescent), are inconsistent. Kelley says:
 > I think the most critical challenge for new demographic inference methods will be to resolve this issue

 - Beichman, Annabel C., Tanya N. Phung, and Kirk E. Lohmueller. 2017. [Comparison of Single Genome and Allele Frequency Data Reveals Discordant Demographic Histories.](http://dx.doi.org/10.1534/g3.117.300259) G3  7 (11): 3605–20.

Some notable papers using the forward diffusion framework:

  - Gutenkunst, Ryan N., Ryan D. Hernandez, Scott H. Williamson, and Carlos D. Bustamante. 2009. [Inferring the Joint Demographic History of Multiple Populations from Multidimensional SNP Frequency Data.](http://dx.doi.org/10.1371/journal.pgen.1000695) _PLoS Genetics_ 5 (10): e1000695.

  - Jouganous, Julien, Will Long, Aaron P. Ragsdale, and Simon Gravel. 2017. [Inferring the Joint Demographic History of Multiple Populations: Beyond the Diffusion Approximation.](http://dx.doi.org/10.1534/genetics.117.200493) _Genetics_ 206 (3): 1549–67.


Some optimization stuff:
- https://en.wikipedia.org/wiki/Penalty_method
- https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
- https://en.wikipedia.org/wiki/Interior-point_method
