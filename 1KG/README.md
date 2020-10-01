1000 Genomes Pipeline
==

Using NYGC 30x call set

Set up environment
--
```bash
conda env create -f env.yml
```

Nexflow Pipeline
--
To run on an SGE cluster:
```bash
nextflow run 1KG.nf -with-report report.html -with-timeline timeline.html -with-dag dag.html
```
