#!usr/bin/bash

cd /net/harris/vol1/project/mushi/1KG
source activate mushi

# linux
n_jobs=`nproc --all`
# n_jobs=1
# mac
# n_jobs=`sysctl -n hw.physicalcpu`

bcfs="/net/harris/vol1/data/phase3_1000genomes/bcfs"
ancs="/net/harris/vol1/data/phase3_1000genomes/human_ancestor_GRCh37_e59"
sample_pops="/net/harris/vol1/data/phase3_1000genomes/integrated_call_samples_v3.20130502.ALL.panel"
k="5"
mutation_rate="1.1e-8"
phastcons="/net/harris/vol1/data/phase3_1000genomes/phastConsElements100way.txt.gz"
repeats="/net/harris/vol1/data/phase3_1000genomes/nestedRepeats.txt.gz"

outdir="/net/harris/vol1/project/mushi/1KG/scons_output_5mer"

# no_exec="--no-exec"

cmd="scons --bcfs=${bcfs} --ancs=${ancs} --mutation_rate=${mutation_rate} --kmer=${k} --phastcons=${phastcons} --repeats=${repeats} --jobs=${n_jobs} --outdir=${outdir} ${debug} ${no_exec} --sample_pops=${sample_pops} ${gl}"
echo $cmd
$cmd
