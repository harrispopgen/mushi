1000 Genomes Pipeline
==

Using NYGC 30x call set

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
nextflow run 1KG.nf -with-report report.html -with-timeline timeline.html -with-dag dag.html
```
