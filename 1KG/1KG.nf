#!/usr/bin/env nextflow

params.hg38_dir = "/net/harris/vol1/data/hg38/"
params.vcf_dir = "/net/harris/vol1/nygc-transfered/"
params.masks = "/net/harris/vol1/data/phase3_1000genomes_supporting/accessible_genome_mask_hg38/pilot_strict_combined.allChr.mask.bed"
params.outgroup_fasta = "/net/harris/vol1/data/panTro6/panTro6.fa"
params.chains = "/net/harris/vol1/data/alignment_chains/hg38ToPanTro6.over.chain.gz"
params.samples = "/net/harris/vol1/data/phase3_1000genomes/integrated_call_samples_v3.20130502.ALL.panel"
params.k = 3

chromosomes = 1..22

ref_fastagz_channel = Channel
  .of (chromosomes)
  .map { [it, file(params.hg38_dir + "chr${it}.fa.gz")] }

Channel
  .of (chromosomes)
  .map { [it, file(params.vcf_dir + "CCDG_13607_B01_GRM_WGS_2019-02-19_chr${it}.recalibrated_variants.vcf.gz")] }
  .into { vcf_channel_1; vcf_channel_2 }

process mask_and_ancestor {

  executor 'sge'
  memory '5 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'ref.fa.gz', 'snps.vcf.gz' from ref_fastagz_channel.join(vcf_channel_1)
  file 'masks.bed' from file(params.masks)
  file 'outgroup.fa' from file(params.outgroup_fasta)
  file 'chain.gz' from file(params.chains)

  output:
  tuple chrom, 'mask.bed' into mask_channel
  tuple chrom, 'ancestor.fa' into ancestor_channel

  """
  zcat -f ref.fa.gz > ref.fa
  grep -P "^chr${chrom}\\t.*strict\$" masks.bed | cut -f1-3 > mask.bed
  bcftools view -T mask.bed -Ou -f PASS -U snps.vcf.gz | mutyper ancestor --bed mask.bed - ref.fa outgroup.fa chain.gz ancestor.fa
  """
}

mask_channel.into{ mask_channel_1; mask_channel_2 }
ancestor_channel.into{ ancestor_channel_1; ancestor_channel_2 }

process masked_size {

  executor 'sge'
  memory '5 GB'
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
  memory '10 MB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  stdin 'masked_size*.tsv' from masked_size_channel.collect()

  output:
  stdout into masked_size_total_channel

  """
  #! /usr/bin/env python

  import sys
  from collections import Counter

  sizes = Counter()
  for line in sys.stdin:
      context, count = line.rstrip().split()
      sizes[context] += int(count)

  for context in sorted(sizes):
      print(f'{context}\t{sizes[context]}')
  """
}

process mutation_types {

  executor 'sge'
  memory '5 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'ancestor.fa', 'snps.vcf.gz' from mask_channel_2.join(ancestor_channel_2).join(vcf_channel_2)
  val k from params.k

  output:
  file 'mutation_types.bcf' into mutation_types_channel

  """
  bcftools view -T mask.bed -m2 -M2 -v snps -c 1:minor -Ou -f PASS -U snps.vcf.gz | bcftools view -g ^miss -Ou | mutyper variants ancestor.fa - --k ${k} | bcftools convert -Ob > mutation_types.bcf
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
  tuple population, 'mutation_types.bcf' from Channel.fromList(populations).combine(mutation_types_channel)
  file 'integrated_call_samples.tsv' from file(params.samples)

  output:
  tuple population, 'ksfs.tsv' into ksfs_channel

  """
  awk '{if(\$2=="${population}"){print \$1}}' integrated_call_samples.tsv > samples.txt
  bcftools view -S samples.txt -c 1:minor -Ou mutation_types.bcf | mutyper ksfs - > ksfs.tsv
  """
}

// // ksfs for each population
// process ksfs_total {
//
//   executor 'sge'
//   memory '5 GB'
//   scratch true
//   conda "${CONDA_PREFIX}/envs/1KG"
//
//   input:
//   tuple population, 'ksfs*.tsv' from ksfs_channel.collate(chromosomes.size())
//
//   output:
//   tuple population, 'ksfs.tsv' into ksfs_total_channel
//
//   """
//   """
// }
