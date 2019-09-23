#!usr/bin/bash

# cd /net/harris/vol1/project/mushi
# source activate mushi

# linux
# n_jobs=`nproc --all`
# mac
n_jobs=`sysctl -n hw.physicalcpu`

bcfs="data.nosync/bcf_files"
ancs="data.nosync/human_ancestor_GRCh37_e59"
sample_pops="data.nosync/integrated_call_samples_v3.20130502.ALL.panel"
pop_flag="--pop=CEU"
k="3"
phastcons="data.nosync/phastConsElements100way.txt.gz"
repeats="data.nosync/nestedRepeats.txt.gz"

outdir="scons_output"

no_exec="--no-exec"

cmd="scons --bcfs=${bcfs} --ancs=${ancs} ${pop_flag} --kmer=${k} --phastcons=${phastcons} --repeats=${repeats} --jobs=${n_jobs} --outdir=${outdir} ${debug} ${no_exec} --sample_pops=${sample_pops} ${gl}"
echo $cmd
$cmd
