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

# all 1KG
pops="--pops=GBR,FIN"

k="3"

mask="../data/phase3_1000genomes/20140520.strict_mask.autosomes.bed"

mushi_cfg="3mer.cfg"

outdir="scons_output_test"

# no_exec="--no-exec"

cmd="scons --mask=${mask} ${pops} --bcfs=${bcfs} --ancs=${ancs} --kmer=${k} --jobs=${n_jobs} --outdir=${outdir} ${debug} ${no_exec} --samples=${samples} --mushi_cfg=${mushi_cfg}"
echo $cmd
$cmd
