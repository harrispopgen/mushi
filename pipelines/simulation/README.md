Simulation Pipeline
==

Set up environment
--
```bash
conda env create -f env.yml
```

You'll also need a [tex installation](https://www.latex-project.org/get/) for plotting functions.

Nexflow Pipeline
--
To run on an SGE cluster:
```bash
nextflow run simulation.nf -with-report report.html -with-timeline timeline.html -with-dag dag.html
```

Plot Notebook
--
The Jupyter notebook `plots.ipynb` generates plots using output from the nextflow pipeline.
Run the notebook within the `simulation` conda environment created in the first step.
