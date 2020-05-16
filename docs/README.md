Update documentation
====================

To rebuild the documentation, install several packages:

Environment:
```bash
conda create -n mushi-docs python=3.7 sphinx pandoc pip clang_osx-64 gsl
conda activate mushi-docs
```

```bash
pip install -r docs/requirements.txt
```

You run at top-level one of the following commands::
```bash
sphinx-build -b html docs docs/_build
```

You can then see the generated documentation in
``docs/_build/index.html``.
