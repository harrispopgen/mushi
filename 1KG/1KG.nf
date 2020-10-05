#!/usr/bin/env nextflow

params.hg38_dir = "/net/harris/vol1/data/hg38/"
params.vcf_dir = "/net/harris/vol1/nygc-transfered/"
params.masks = "/net/harris/vol1/data/phase3_1000genomes_supporting/accessible_genome_mask_hg38/pilot_strict_combined.allChr.mask.bed"
params.outgroup_fasta = "/net/harris/vol1/data/panTro6/panTro6.fa"
params.chains = "/net/harris/vol1/data/alignment_chains/hg38ToPanTro6.over.chain.gz"
params.samples = "/net/harris/vol1/data/phase3_1000genomes/integrated_call_samples_v3.20130502.ALL.panel"
params.outdir = "output"
params.k = 3

chromosomes = 1..22

ref_fastagz_channel = Channel
  .of (chromosomes)
  .map { [it, file(params.hg38_dir + "chr${it}.fa.gz")] }

Channel
  .of (chromosomes)
  .map { [it, file(params.vcf_dir + "CCDG_13607_B01_GRM_WGS_2019-02-19_chr${it}.recalibrated_variants.vcf.gz"), file(params.vcf_dir + "CCDG_13607_B01_GRM_WGS_2019-02-19_chr${it}.recalibrated_variants.vcf.gz.tbi")] }
  .into { vcf_channel_1; vcf_channel_2 }

process mask_and_ancestor {

  executor 'sge'
  memory '10 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'ref.fa.gz', 'snps.vcf.gz', 'snps.vcf.gz.tbi' from ref_fastagz_channel.join(vcf_channel_1)
  file 'masks.bed' from file(params.masks)
  file 'outgroup.fa' from file(params.outgroup_fasta)
  file 'chain.gz' from file(params.chains)

  output:
  tuple chrom, 'mask.bed' into mask_channel
  tuple chrom, 'ancestor.fa' into ancestor_channel

  """
  zcat -f ref.fa.gz > ref.fa
  grep -P "^chr${chrom}\\t.*strict\$" masks.bed | cut -f1-3 > mask.bed
  bcftools view -R mask.bed -Ou -f PASS -U snps.vcf.gz | mutyper ancestor --bed mask.bed - ref.fa outgroup.fa chain.gz ancestor.fa
  """
}

mask_channel.into{ mask_channel_1; mask_channel_2 }
ancestor_channel.into{ ancestor_channel_1; ancestor_channel_2 }

process masked_size {

  executor 'sge'
  memory '10 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed' from mask_channel_1
  tuple chrom, 'ancestor.fa' from ancestor_channel_1
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
  memory '10 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'ancestor.fa', 'snps.vcf.gz', 'snps.vcf.gz.tbi' from mask_channel_2.join(ancestor_channel_2).join(vcf_channel_2)
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
  file 'ksfs.tsv' into ksfs_total_channel

  """
  #! /usr/bin/env python

  import glob
  import pandas as pd

  sum(pd.read_csv(file, sep='\t', index_col=0)
      for file in glob.glob('ksfs*')).to_csv('ksfs.tsv', sep='\t')
  """
}
