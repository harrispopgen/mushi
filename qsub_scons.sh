#!usr/bin/bash

cd /net/harris/vol1/project/europulse
source activate europulse


n_jobs=`nproc --all`
ref_dir="/net/harris/vol1/data/phase3_1000genomes/bcfs"
sample_pops="/net/harris/vol1/data/phase3_1000genomes/integrated_call_samples_v3.20130502.ALL.panel"
# pop_flag="--pop=CEU"
hg19="/net/harris/vol1/project/europulse/data/hg19NoChr.fa"
hg19anc="/net/harris/vol1/project/europulse/data/hg19ancNoChr.fa"
metadata="/net/gs/vol1/home/wsdewitt/kelley_home/project/europulse/data/kelley_bamInfo.xlsx"
bam_dir="/net/harris/vol1/project/europulse/data/Martin_2018.06.04/"
phastcons="/net/harris/vol1/project/europulse/data/phastConsElements100way.txt.gz"
repeats="/net/harris/vol1/project/europulse/data/nestedRepeats.txt.gz"

# outdir="/net/harris/vol1/project/europulse/scons_output"
outdir="/net/harris/vol1/project/europulse/scons_output_debug"
debug="--debug_mode"

# no_exec="--no-exec"

# set this if you want genotype likelihood fanciness
gl="--gl"

cmd="scons --metadata=${metadata} --bam_dir=${bam_dir} --ref=${ref_dir} ${pop_flag} --hg19=${hg19} --hg19_ancestral=${hg19anc} --phastcons=${phastcons} --repeats=${repeats} --jobs=${n_jobs} --outdir=${outdir} ${debug} ${no_exec} --sample_pops=${sample_pops} ${gl}"
echo $cmd
$cmd
