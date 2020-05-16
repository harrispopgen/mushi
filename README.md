![](docs/_static/logo.png)

A Python package for: [__mu__]tation [__s__]pectrum [__h__]istory [__i__]nference

Pairs well with the package [`mutyper`](https://github.com/harrispopgen/mutyper), which assigns mutation types to SNPs in VCF files.

Installation
---
- Python 3.7
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

- Additional dependencies are needed to run the [Jupyter notebooks](docs/notebooks) or the [1000 Genomes pipeline](1KG). A complete [Conda](https://docs.conda.io/en/latest/) environment (including `mushi`) can created with
```bash
$ conda env create -f env.yml
```
and then activated with
```bash
$ conda activate mushi
```


Citing `mushi`
---

No preprint yet, so for now check out our [TAGC2020 poster](https://tagc2020.figshare.com/articles/Joint_nonparametric_coalescent_inference_of_mutation_spectrum_history_and_demography/12149790):
```
@misc{dewitt_deckerharris_harris_2020,
      title={Joint nonparametric coalescent inference of mutation spectrum history and demography},
      url={https://tagc2020.figshare.com/articles/Joint_nonparametric_coalescent_inference_of_mutation_spectrum_history_and_demography/12149790/1},
      DOI={10.6084/m9.figshare.12149790.v1},
      abstractNote={PDF for poster 1232B},
      publisher={figshare},
      author={DeWitt, William S. and Decker Harris, Kameron and Harris, Kelley}, year={2020}, month={Apr} }
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


Example data and notebooks
---
- [Example 3-SFS data](docs/example_data) for each 1000 Genomes population are in, for use with notebooks.
- Jupyter [notebooks](docs/notebooks)

[TODO: document 1KG pipeline]
