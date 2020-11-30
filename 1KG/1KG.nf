#!/usr/bin/env nextflow

params.vcf_dir = "/net/harris/vol1/nygc-transfered/"
params.mask = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/working/20160622_genome_mask_GRCh38/StrictMask/20160622.allChr.mask.bed"
params.ancestor = "ftp://ftp.ensembl.org/pub/release-100/fasta/ancestral_alleles/homo_sapiens_ancestor_GRCh38.tar.gz"
params.samples = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
params.outdir = "output"
params.k = 3

chromosomes = 1..22

vcf_channel = Channel
  .of (chromosomes)
  .map { [it,
          file(params.vcf_dir + "CCDG_13607_B01_GRM_WGS_2019-02-19_chr${it}.recalibrated_variants.vcf.gz"),
          file(params.vcf_dir + "CCDG_13607_B01_GRM_WGS_2019-02-19_chr${it}.recalibrated_variants.vcf.gz.tbi")] }

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

// mutation types for each chromosome vcf
process ksfs {

  executor 'sge'
  memory '200 MB'
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
  cmd="bcftools view -S all_samples.txt -c 1:minor -R mask.bed -m2 -M2 -v snps -f PASS -g ^miss -I -Ou snps.vcf.gz | mutyper variants ancestor.fa - --strict --k ${k} | tee "
  for pop in `tail -n +2 integrated_call_samples.tsv | cut -f2 | sort | uniq`; do
    awk '{if(\$2=="\${pop}"){print \$1}}' integrated_call_samples.tsv > \${pop}_samples.txt
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
  publishDir "$params.outdir/${population}"

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

process mushi {

  executor 'sge'
  memory '500 MB'
  time '1h'
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
