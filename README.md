![](logo.png)

A Python package for: [__mu__]tation [__s__]pectrum [__h__]istory [__i__]nference

Pairs well with the package [`mutyper`](https://github.com/harrispopgen/mutyper), which assigns mutation types to SNPs in VCF files.

Installation
---

- Basic install with pip
```bash
$ pip install git+https://github.com/harrispopgen/mushi
```

- Developer installation: add `-e` for editable (clones repo to `./src/mushi`)
```bash
$ pip install -e git+https://github.com/harrispopgen/mushi#egg=mushi
```

- For LaTeX rendering in `mushi` plotting functions, you may also need to install dependencies for the [`matplotlib` `usetex`](https://matplotlib.org/tutorials/text/usetex.html) option:
>Matplotlib's LaTeX support requires a working [LaTeX](http://www.tug.org/) installation, [dvipng](http://www.nongnu.org/dvipng/)
(which may be included with your LaTeX installation), and [Ghostscript](https://ghostscript.com/)
(GPL Ghostscript 9.0 or later is required). The executables for these
external dependencies must all be located on your [`PATH`](https://matplotlib.org/faq/environment_variables_faq.html#envvar-PATH).

- Additional dependencies are needed to run the [Jupyter notebooks](notebooks) or the [1000 Genomes pipeline](1KG). A complete [Conda](https://docs.conda.io/en/latest/) environment (including `mushi`) can created with
```bash
$ conda env create -f env.yml
```
and then activated with
```bash
$ conda activate mushi
```


Citing `mushi`
---

No preprint yet, so for now:
```
@software{mushi2020github,
  author = {William DeWitt},
  title = {mushi: mutation spectrum history inference},
  url = {https://github.com/harrispopgen/mushi},
  version = {1.0},
  year = {2020},
}
```

Command line interface
---

TODO


Python API
---

See API documentation at [???]

- [`ksfs.py`](mushi/ksfs.py): Class implementing the coalescent model of the expected sample frequency spectrum
- [`histories.py`](mushi/histories.py): Classes for demographic history and mutation spectrum history objects.
- [`optimization.py`](mushi/optimization.py): proximal gradient algorithms.
- [`utils.py`](mushi/utils.py): utility functions.


Example data
---
3-SFS data for each 1000 Genomes population are in [`example_data`](example_data), for use with notebooks.

Jupyter [notebooks](notebooks)
---

- [`demo-mushi.ipynb`](notebooks/demo-mushi.ipynb): Demostration of API usage and interface with `mutyper` output.
- [`simulation_simple.ipynb`](notebooks/simulation_simple.ipynb): Jupyter notebook exploring reconstruction on k-SFS simulated under the coalescent forward model.
- [`simulation.ipynb`](notebooks/simulation.ipynb): Jupyter notebook exploring reconstruction on k-SFS simulated with `msprime` and `stdpopsim`.
- [`L-spectrum.ipynb`](notebooks/L-spectrum.ipynb): Spectral analysis of the L matrix defined in the text.

[TODO: document 1KG pipeline]
