#!/usr/bin/env nextflow

params.vcf_dir = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/"
params.mask = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/working/20160622_genome_mask_GRCh38/StrictMask/20160622.allChr.mask.bed"
params.ancestor = "ftp://ftp.ensembl.org/pub/release-100/fasta/ancestral_alleles/homo_sapiens_ancestor_GRCh38.tar.gz"
params.samples = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
params.outdir = "output"
params.k = 3

chromosomes = 1..22

vcf_channel = Channel
  .of (chromosomes)
  .map { [it,
          file(params.vcf_dir + "CCDG_14151_B01_GRM_WGS_2020-08-05_chr${it}.filtered.shapeit2-duohmm-phased.vcf.gz"),
          file(params.vcf_dir + "CCDG_14151_B01_GRM_WGS_2020-08-05_chr${it}.filtered.shapeit2-duohmm-phased.vcf.gz.tbi")] }

process mask {

  executor 'sge'
  memory '10 MB'
  time '10m'
  scratch true

  input:
  path 'mask.allchr.bed' from params.mask
  each chromosome from chromosomes

  output:
  tuple chromosome, 'mask.bed' into mask_channel

  """
  grep -P "^chr${chromosome}\\t" mask.allchr.bed | cut -f1-3 > mask.bed
  """
}

process ancestor {

  executor 'sge'
  memory '10 MB'
  time '10m'
  scratch true

  input:
  path 'homo_sapiens_ancestor_GRCh38.tar.gz' from params.ancestor
  each chromosome from chromosomes

  output:
  tuple chromosome, 'ancestor.fa' into ancestor_channel

  """
  tar -zxvf homo_sapiens_ancestor_GRCh38.tar.gz homo_sapiens_ancestor_GRCh38/homo_sapiens_ancestor_${chromosome}.fa
  echo ">chr${chromosome}" > ancestor.fa
  tail -n +2 homo_sapiens_ancestor_GRCh38/homo_sapiens_ancestor_${chromosome}.fa >> ancestor.fa
  """
}

mask_channel.into{ mask_channel_1; mask_channel_2 }
ancestor_channel.into{ ancestor_channel_1; ancestor_channel_2 }

process masked_size {

  executor 'sge'
  memory '100 MB'
  time '10h'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'ancestor.fa' from mask_channel_1.join(ancestor_channel_1)
  val k from params.k

  output:
  file 'masked_size.tsv' into masked_size_channel

  """
  mutyper targets ancestor.fa --strict --k ${k} --bed mask.bed > masked_size.tsv
  """
}

process masked_size_total {

  executor 'sge'
  memory '100 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir params.outdir, mode: 'copy'

  input:
  file 'masked_size' from masked_size_channel.collect()

  output:
  file 'masked_size.tsv' into masked_size_total_channel

  """
  #! /usr/bin/env python

  import glob
  import pandas as pd

  sum(pd.read_csv(file, sep='\t', index_col=0, header=None, squeeze=True)
      for file in glob.glob('masked_size*')).to_csv('masked_size.tsv', sep='\t', header=False)
  """
}

// mutation types for each chromosome vcf
// this is overcomplicated with process substitution to avoid writing large files
process ksfs {

  executor 'sge'
  memory '500 MB'
  penv 'serial' // UWGS parallel environment
  cpus 27 // 26 1KG populations
  time '1d'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'ancestor.fa', 'snps.vcf.gz', 'snps.vcf.gz.tbi' from mask_channel_2.join(ancestor_channel_2).join(vcf_channel)
  path 'integrated_call_samples.tsv' from params.samples
  val k from params.k

  output:
  file '*.ksfs.tsv' into ksfs_channel mode flatten

  """
  tail -n +2 integrated_call_samples.tsv | cut -f1 > all_samples.txt
  cmd="bcftools view -S all_samples.txt -c 1:minor -R mask.bed -m2 -M2 -v snps -f PASS -Ou snps.vcf.gz | bcftools view -g ^miss -Ou | mutyper variants ancestor.fa - --strict --k ${k} | tee "
  for pop in `tail -n +2 integrated_call_samples.tsv | cut -f2 | sort | uniq`; do
    grep \${pop} integrated_call_samples.tsv | cut -f1 > \${pop}_samples.txt
    cmd=\$cmd" >(bcftools view -S \${pop}_samples.txt -c 1:minor -G -Ou | mutyper ksfs - > \${pop}.${chrom}.ksfs.tsv) "
  done
  cmd=\$cmd" > /dev/null"
  eval \$cmd
  """
}


// ksfs for each population
process ksfs_total {

  executor 'sge'
  memory '100 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple population, 'ksfs' from ksfs_channel.map{file -> tuple(file.simpleName, file)}.groupTuple(size: chromosomes.size())

  output:
  tuple population, 'ksfs.tsv' into ksfs_total_channel

  """
  #! /usr/bin/env python

  import glob
  import pandas as pd

  sum(pd.read_csv(file, sep='\t', index_col=0)
      for file in glob.glob('ksfs*')).to_csv('ksfs.tsv', sep='\t')
  """
}

alpha_tv = [0] + (1..2).by(0.2).collect { 10**it }
alpha_spline = [0] + (1..2).by(1).collect { 10**it }
alpha_ridge = 1e-4

process mushi {

  executor 'sge'
  memory '500 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/${alpha_tv}_${alpha_spline}/${population}", mode: 'copy'

  input:
  tuple population, 'ksf.tsv' from ksfs_total_channel
  file 'masked_size.tsv' from masked_size_total_channel
  each alpha_tv from alpha_tv
  each alpha_spline from alpha_spline
  each alpha_ridge from alpha_ridge

  output:
  file 'sfs.pdf' into sfs_plot

  """
  #! /usr/bin/env python

  import mushi
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt

  ksfs = mushi.kSFS(file='ksf.tsv')

  # sorts the columns of the ksfs
  sorted_triplets = [f'{a5}{a}{a3}>{a5}{d}{a3}' for a in 'AC'
                     for d in 'ACGT' if d != a
                     for a5 in 'ACGT' for a3 in 'ACGT']
  foo, bar = ksfs.mutation_types.reindex(sorted_triplets)
  ksfs.mutation_types = foo
  ksfs.X = ksfs.X[:, bar]

  masked_genome_size = pd.read_csv('masked_size.tsv', sep='\t', header=None, index_col=0, names=('count',))

  change_points = np.logspace(np.log10(1), np.log10(200000), 200)

  u = 1.25e-8
  mu0 = u * masked_genome_size['count'].sum()

  t_gen = 29

  clip_low = 0
  clip_high = 10
  freq_mask = np.array([True if (clip_low <= i < ksfs.n - clip_high - 1) else False
                        for i in range(ksfs.n - 1)])

  ksfs.infer_history(change_points, mu0,
                     infer_mu=False, loss='prf', mask=freq_mask,
                     alpha_tv=${alpha_tv}, alpha_spline=${alpha_spline}, alpha_ridge=${alpha_ridge},
                     tol=1e-10, max_iter=1000)

  fig = plt.figure(figsize=(6, 3))
  plt.subplot(1, 2, 1)
  ksfs.plot_total(kwargs=dict(ls='', marker='o', ms=5, mfc='none'),
                  line_kwargs=dict(ls=':', marker='.', ms=3, lw=1),
                  fill_kwargs=dict(alpha=0))
  plt.xscale('log')
  plt.yscale('log')
  plt.subplot(1, 2, 2)
  ksfs.eta.plot(t_gen=t_gen, lw=3)
  plt.xlim([1e3, 1e6])
  plt.tight_layout()
  plt.savefig('sfs.pdf')


  """
}
