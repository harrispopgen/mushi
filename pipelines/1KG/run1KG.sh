#!/bin/bash
#$ -cwd

nextflow run 1KG.nf -with-report report.html -with-timeline timeline.html -with-dag dag.html
# -resume
