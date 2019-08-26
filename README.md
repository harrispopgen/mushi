mushi
====

[mu]tation [s]pectrum [h]istory [i]nference

Dope bible, manga, and hardcore punk things that share this name: https://en.wikipedia.org/wiki/Mushi

Dependencies
---
  - python3.7+
  - numpy
  - scipy
  - matplotlib
  - jupyter
  - scons
  - [prox-tv](https://github.com/albarji/proxTV)

Writing
---
Protomanuscript forming here:
- LaTeX: [manuscript/main.tex](manuscript/main.tex)
- References: [manuscript/refs.bib](manuscript/refs.bib)
- build `_build/main.pdf` by issuing the following command from within the `manuscript` subdirectory
```bash
$ scons
```

Code
---
- [`dement.py`](dement.py): Class implementing the coalescent model of the expected sample frequency spectrum, as described in Rosen et al., and adding the Poisson random field for a full generative model.
- [`regularization.ipynb`](regularization.ipynb): Jupyter notebook exploring penalized likelihood optimization.
- [`A_matrix.ipynb`](A_matrix.ipynb): Investigating properties of Polanski and Kimmel's A_n matrix.


References
---

(more exhaustive list in [manuscript/refs.bib]())

#### SFS Model
The coalescent model we're using for the site-frequency spectrum.

- Rosen, Zvi, Anand Bhaskar, Sebastien Roch, and Yun S. Song. 2018. [Geometry of the Sample Frequency Spectrum and the Perils of Demographic Inference.](http://dx.doi.org/10.1534/genetics.118.300733) _Genetics_, July, genetics.300733.2018.

- Polanski, A., and M. Kimmel. 2003. [New Explicit Expressions for Relative Frequencies of Single-Nucleotide Polymorphisms with Application to Statistical Inference on Population Growth.](http://www.genetics.org/cgi/pmidlookup?view=long&pmid=14504247) _Genetics_ 165 (1): 427–36.

- Bhaskar, Anand, Y. X. Rachel Wang, and Yun S. Song. 2015. [Efficient Inference of Population Size Histories and Locus-Specific Mutation Rates from Large-Sample Genomic Variation Data.](http://dx.doi.org/10.1101/gr.178756.114) Genome Research 25 (2): 268–79.

#### Entropy-based regularization

- Regularization with a Bregman divergence from the last iterate is known as [Mirror Descent](https://blogs.princeton.edu/imabandit/2013/04/16/orf523-mirror-descent-part-iii/).
- Will knows it as the "Maximum Entropy Method" from his old physics paper.


#### Forward diffusion framework:

- Ragsdale, Aaron P., Claudia Moreau, and Simon Gravel. 2018. [Genomic Inference Using Diffusion Models and the Allele Frequency Spectrum.](http://dx.doi.org/10.1016/j.gde.2018.10.001) Current Opinion in Genetics & Development 53 (October): 140–47.

- Gutenkunst, Ryan N., Ryan D. Hernandez, Scott H. Williamson, and Carlos D. Bustamante. 2009. [Inferring the Joint Demographic History of Multiple Populations from Multidimensional SNP Frequency Data.](http://dx.doi.org/10.1371/journal.pgen.1000695) _PLoS Genetics_ 5 (10): e1000695.

- Jouganous, Julien, Will Long, Aaron P. Ragsdale, and Simon Gravel. 2017. [Inferring the Joint Demographic History of Multiple Populations: Beyond the Diffusion Approximation.](http://dx.doi.org/10.1534/genetics.117.200493) _Genetics_ 206 (3): 1549–67.


#### SMC++ (Yun's group):
There's some regularization / spline regression we might want to look at here
- Terhorst, Jonathan, John A. Kamm, and Yun S. Song. 2017. [Robust and Scalable Inference of Population History from Hundreds of Unphased Whole Genomes.](http://dx.doi.org/10.1038/ng.3748) Nature Genetics 49 (2): 303–9.

#### Optimization:
- https://en.wikipedia.org/wiki/Penalty_method
- https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
- https://en.wikipedia.org/wiki/Interior-point_method

#### SFS Vs LD discordance
This paper shows that SFS-based methods and LD-based methods (sequential Markov coalescent), are inconsistent. Kelley says:
 > I think the most critical challenge for new demographic inference methods will be to resolve this issue

- Beichman, Annabel C., Tanya N. Phung, and Kirk E. Lohmueller. 2017. [Comparison of Single Genome and Allele Frequency Data Reveals Discordant Demographic Histories.](http://dx.doi.org/10.1534/g3.117.300259) G3  7 (11): 3605–20.
