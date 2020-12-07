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
