#!usr/bin/bash

# cd /net/harris/vol1/project/mushi/pipeline_1KG
# source activate mushi

# linux
# n_jobs=`nproc --all`
# mac
n_jobs=`sysctl -n hw.physicalcpu`

bcfs="../data/phase3_1000genomes/bcfs"
ancs="../data/human_ancestor_GRCh37_e59"
samples="../data/phase3_1000genomes/integrated_call_samples_v3.20130502.ALL.panel"
mask="../data/phase3_1000genomes/20140520.pilot_mask.autosomes.bed"

# restrict to pops
# pops="--pops=GBR,FIN"

# restrict to one chromosome
# chrom="--chrom=22"

# kmer
k="3"

outdir="scons_output_pilot"

# no_exec="--no-exec"

cmd="scons --mask=${mask} ${pops} ${chrom} --bcfs=${bcfs} --ancs=${ancs} --kmer=${k} --jobs=${n_jobs} --outdir=${outdir} ${debug} ${no_exec} --samples=${samples}"
echo $cmd
$cmd
