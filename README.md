mushi
====

[__mu__]tation [__s__]pectrum [__h__]istory [__i__]nference

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
  - [msprime](https://msprime.readthedocs.io)

Manuscript
---
- TeX: [manuscript/main.tex](manuscript/main.tex)
- References: [manuscript/refs.bib](manuscript/refs.bib)
- build `_build/main.pdf` by issuing the following command from within the `manuscript` subdirectory
```bash
$ scons
```

Code
---
- [`mushi.py`](mushi.py): Class implementing the coalescent model of the expected sample frequency spectrum, as described in the text.
- [`regularization.ipynb`](regularization.ipynb): Jupyter notebook exploring penalized likelihood optimization.
- [`C_matrix.ipynb`](C_matrix.ipynb): Investigating properties of the C matrix defined in the text.
