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

For LaTeX rendering in plotting functions, you may also need to install additional dependencies for the [`matplotlib` `usetex`](https://matplotlib.org/tutorials/text/usetex.html) option:
>Matplotlib's LaTeX support requires a working [LaTeX](http://www.tug.org/) installation, [dvipng](http://www.nongnu.org/dvipng/)
(which may be included with your LaTeX installation), and [Ghostscript](https://ghostscript.com/)
(GPL Ghostscript 9.0 or later is required). The executables for these
external dependencies must all be located on your [`PATH`](https://matplotlib.org/faq/environment_variables_faq.html#envvar-PATH).

Code
---
- [`mushi.py`](mushi.py): Class implementing the coalescent model of the expected sample frequency spectrum, as described in [TODO: citation when it exists].
- [`tcc-pulse-timing.ipynb`](tcc-pulse-timing.ipynb): Jupyter notebook for inference of European demographies and TCC>TTC pulse, using example data.
- [`simulation_simple.ipynb`](simulation_simple.ipynb): Jupyter notebook exploring reconstruction on SFS simulated under the mushi forward model.
- [`simulation.ipynb`](simulation.ipynb): Jupyter notebook exploring reconstruction on SFS simulated with `msprime` and `stdpopsim`.
- [`L-spectrum.ipynb`](L-spectrum.ipynb): Spectral analysis of the L matrix defined in the text.
- [`histories.py`](histories.py): Classes for demographic history and mutation spectrum history objects.
- [`optimization.py`](optimization.py): proximal gradient algorithms.
- [`utils.py`](utils.py): utility functions.

[TODO: document 1KG pipeline, mutyper subpackage]

Example data
---
3-SFS data for each 1000 Genomes population are in [`example_data`](example_data), for use with notebooks.
