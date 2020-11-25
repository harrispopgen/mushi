#!/usr/bin/env nextflow

params.vcf_dir = "/net/harris/vol1/nygc-transfered/"
params.masks = "/net/harris/vol1/data/phase3_1000genomes_supporting/accessible_genome_mask_hg38/pilot_strict_combined.allChr.mask.bed"
params.ancestor = "/net/harris/vol1/data/homo_sapiens_ancestor_GRCh38/"
params.samples = "/net/harris/vol1/data/phase3_1000genomes/integrated_call_samples_v3.20130502.ALL.panel"
params.outdir = "output"
params.k = 3

chromosomes = 1..22

vcf_channel = Channel
  .of (chromosomes)
  .map { [it, file(params.vcf_dir + "CCDG_13607_B01_GRM_WGS_2019-02-19_chr${it}.recalibrated_variants.vcf.gz"), file(params.vcf_dir + "CCDG_13607_B01_GRM_WGS_2019-02-19_chr${it}.recalibrated_variants.vcf.gz.tbi")] }

process mask {

  executor 'sge'
  memory '10 MB'
  scratch true

  input:
  file 'masks.bed' from file(params.masks)
  each chromosome from chromosomes


  output:
  tuple chromosome, 'mask.bed' into mask_channel

  """
  grep -P "^chr${chromosome}\\t.*strict\$" masks.bed | cut -f1-3 > mask.bed
  """
}

ancestor_channel = Channel
  .of (chromosomes)
  .map { [it, file(params.ancestor + "homo_sapiens_ancestor_${it}.fa")] }

process ancestor_chr {

  executor 'sge'
  memory '10 MB'
  scratch true

  input:
  tuple chromosome, 'ancestor.fa' from ancestor_channel


  output:
  tuple chromosome, 'ancestor.chr.fa' into ancestor_chr_channel

  """
  echo ">chr${chromosome}" > ancestor.chr.fa
  tail -n +2 ancestor.fa >> ancestor.chr.fa
  """
}

mask_channel.into{ mask_channel_1; mask_channel_2 }
ancestor_chr_channel.into{ ancestor_chr_channel_1; ancestor_chr_channel_2 }

process masked_size {

  executor 'sge'
  memory '5 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'ancestor.fa' from mask_channel_1.join(ancestor_chr_channel_1)
  val k from params.k

  output:
  file 'masked_size.tsv' into masked_size_channel

  """
  mutyper targets ancestor.fa --k ${k} --bed mask.bed > masked_size.tsv
  """
}

process masked_size_total {

  executor 'sge'
  memory '1 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir params.outdir

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

populations = ["ACB", "ASW", "BEB", "CDX", "CEU", "CHB", "CHS", "CLM", "ESN", "FIN", "GBR", "GIH", "GWD", "IBS", "ITU", "JPT", "KHV", "LWK", "MSL", "MXL", "PEL", "PJL", "PUR", "STU", "TSI", "YRI"]

// ksfs for each population X chromosome
process ksfs {

  executor 'sge'
  memory '5 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'ancestor.fa', 'snps.vcf.gz', 'snps.vcf.gz.tbi' from mask_channel_2.join(ancestor_chr_channel_2).join(vcf_channel)
  each population from populations
  file 'integrated_call_samples.tsv' from file(params.samples)
  val k from params.k

  output:
  tuple population, 'ksfs.tsv' into ksfs_channel

  """
  awk '{if(\$2=="${population}"){print \$1}}' integrated_call_samples.tsv > samples.txt
  bcftools view -R mask.bed -m2 -M2 -v snps -Ou -f PASS -g ^miss snps.vcf.gz | bcftools view -S samples.txt -c 1:minor -Ou | mutyper variants ancestor.fa - --k ${k} | mutyper ksfs - > ksfs.tsv
  """
}

// ksfs for each population
process ksfs_total {

  executor 'sge'
  memory '1 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/${population}"

  input:
  tuple population, 'ksfs' from ksfs_channel.groupTuple(size: chromosomes.size())

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

process mushi {

  executor 'sge'
  memory '2 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/${population}"

  input:
  tuple population, 'ksf.tsv' from ksfs_total_channel
  file 'masked_size.tsv' from masked_size_total_channel

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

  ksfs.infer_history(change_points, mu0,
                     infer_mu=False, loss='prf',
                     alpha_tv=1e2, alpha_spline=3e3, alpha_ridge=1e-4,
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
