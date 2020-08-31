#!usr/bin/bash

source activate 1KG

# linux
n_jobs=`nproc --all`
# mac
# n_jobs=`sysctl -n hw.physicalcpu`

home="/net/harris/vol1"

cd $home/project/mushi/1KG

vcfs="$home/nygc-transfered"
reference="$home/data/hg38"
outgroup="$home/data/panTro6/panTro6.fa"
chains="$home/data/alignment_chains/hg38ToPanTro6.over.chain.gz"
mask="$home/data/phase3_1000genomes_supporting/accessible_genome_mask_hg38/pilot_strict_combined.allChr.mask.bed"
samples="$home/data/phase3_1000genomes/integrated_call_samples_v3.20130502.ALL.panel"

# restrict to pops
pops="--pops=GBR,FIN"

# restrict to one chromosome
chrom="--chrom=chr22"

# kmer
k="3"

outdir="scons_output"

no_exec="--no-exec"

cmd="scons --mask=${mask} ${pops} ${chrom} --vcfs=${vcfs} --reference=${reference} --outgroup=${outgroup} --chains=${chains} --kmer=${k} --jobs=${n_jobs} --outdir=${outdir} ${no_exec} --samples=${samples}"
echo $cmd
$cmd
