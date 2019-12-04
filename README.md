mushi
====

[__mu__]tation [__s__]pectrum [__h__]istory [__i__]nference

Dope bible, manga, and hardcore punk things that share this name: https://en.wikipedia.org/wiki/Mushi

Dependencies
---

Be sure to clone using the `--recurse-submodules` flag to get the `stdpopsim` dependency.

Dependences are listed in [env.yml](). You can set up a [conda](https://docs.conda.io/en/latest/) environment with
```bash
$ conda env create -f env.yml
```
and activate your new environment with
```bash
$ conda activate mushi
```

Code
---
- [`mushi.py`](mushi.py): Class implementing the coalescent model of the expected sample frequency spectrum, as described in the text.
- [`test.ipynb`](test.ipynb): Jupyter notebook exploring reconstruction on SFS simulated under the mushi forward model.
- [`test-msprime.ipynb`](test-msprime.ipynb): Jupyter notebook exploring reconstruction on SFS simulated with msprime.
- [`L_spectrum.ipynb`](L_matrix.ipynb): Investigating properties of the L matrix defined in the text.
