#!usr/bin/bash

# cd /net/harris/vol1/project/mushi/1KG
# source activate mushi

# linux
# n_jobs=`nproc --all`
# n_jobs=1
# mac
n_jobs=`sysctl -n hw.physicalcpu`

bcfs="../data/phase3_1000genomes/bcfs"
ancs="../data/human_ancestor_GRCh37_e59"

sample_pops="../data/phase3_1000genomes/integrated_call_samples_v3.20130502.ALL.panel"

# # superpopulations
# pops="AFR,AMR,EAS,EUR,SAS"

# africans
pops="YRI,LWK,GWD,MSL,ESN,ASW,ACB"

k="3"

phastcons="../data/phastConsElements100way.txt.gz"
repeats="../data/nestedRepeats.txt.gz"

# outdir="scons_output"
outdir="scons_output_AFR"

# track='--track=../data/20161129_enhancer_states.txt.4567.bed'
# comp='--track_complement'

# no_exec="--no-exec"

cmd="scons ${track} ${comp} --pops=${pops} --bcfs=${bcfs} --ancs=${ancs} --kmer=${k} --phastcons=${phastcons} --repeats=${repeats} --jobs=${n_jobs} --outdir=${outdir} ${debug} ${no_exec} --sample_pops=${sample_pops} ${gl}"
echo $cmd
$cmd
