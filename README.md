mushi
====

[__mu__]tation [__s__]pectrum [__h__]istory [__i__]nference

Dope bible, manga, and hardcore punk things that share this name: https://en.wikipedia.org/wiki/Mushi

Dependencies
---
Dependences are listed in [env.yml](). You can set up a conda environment with
```bash
conda env create -f env.yml
```
and activate your new environment with
```bash
conda activate mushi
```

Be sure to clone using the `--recurse-submodules` flag to get the `stdpopsim` dependency.

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
- [`test.ipynb`](test.ipynb): Jupyter notebook exploring reconstruction on SFS simulated under the mushi forward model.
- [`test-msprime.ipynb`](test-msprime.ipynb): Jupyter notebook exploring reconstruction on SFS simulated with msprime.
- [`L_spectrum.ipynb`](L_matrix.ipynb): Investigating properties of the L matrix defined in the text.
