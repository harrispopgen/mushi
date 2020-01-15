mushi
====

[__mu__]tation [__s__]pectrum [__h__]istory [__i__]nference

Dope bible, manga, and hardcore punk things that share this name: https://en.wikipedia.org/wiki/Mushi

Dependencies
---

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
- [`mushi.py`](mushi.py): Class implementing the coalescent model of the expected sample frequency spectrum, as described in [TODO: citation when it exists].
- [`tcc-pulse-timing.ipynb`](tcc-pulse-timing.ipynb): Jupyter notebook for inference of European demographies and TCC>TTC pulse, using example data.
- [`simulation.ipynb`](simulation.ipynb): Jupyter notebook exploring reconstruction on SFS simulated under the mushi forward model.
- [`test-msprime.ipynb`](test-msprime.ipynb): Jupyter notebook exploring reconstruction on SFS simulated with msprime.
- [`L-spectrum.ipynb`](L-spectrum.ipynb): Investigating properties of the L matrix defined in the text.
- [`histories.py`](histories.py): Classes for demographic history and mutation spectrum history objects.
- [`utils.py`](utils.py): utility function, including proximal gradient algorithms.

[TODO: document 1KG pipeline]

Example data
---
3-SFS data for each European population are in [`example_data`](example_data), for use with [`tcc-pulse-timing.ipynb`](tcc-pulse-timing.ipynb) notebook.
