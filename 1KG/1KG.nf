#!/usr/bin/env nextflow

params.vcf_dir = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20181203_biallelic_SNV/"
params.mask = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/working/20160622_genome_mask_GRCh38/StrictMask/20160622.allChr.mask.bed"
params.ancestor = "ftp://ftp.ensembl.org/pub/release-100/fasta/ancestral_alleles/homo_sapiens_ancestor_GRCh38.tar.gz"
params.samples = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
params.outdir = "output"
params.k = 3

chromosomes = 1..22

vcf_ch = Channel
  .of (chromosomes)
  .map { [it,
          file(params.vcf_dir + "ALL.chr${it}.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz"),
          file(params.vcf_dir + "ALL.chr${it}.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz.tbi")] }

process mask {

  executor 'sge'
  memory '10 MB'
  time '10m'
  scratch true

  input:
  path 'mask.allchr.bed' from params.mask
  each chromosome from chromosomes

  output:
  tuple chromosome, 'mask.bed' into mask_ch

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
  tuple chromosome, 'ancestor.fa' into ancestor_ch

  """
  tar -zxvf homo_sapiens_ancestor_GRCh38.tar.gz homo_sapiens_ancestor_GRCh38/homo_sapiens_ancestor_${chromosome}.fa
  echo ">chr${chromosome}" > ancestor.fa
  tail -n +2 homo_sapiens_ancestor_GRCh38/homo_sapiens_ancestor_${chromosome}.fa >> ancestor.fa
  """
}

mask_ch.into{ mask_ch_1; mask_ch_2 }
ancestor_ch.into{ ancestor_ch_1; ancestor_ch_2 }

process masked_size {

  executor 'sge'
  memory '100 MB'
  time '10h'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'ancestor.fa' from mask_ch_1.join(ancestor_ch_1)
  val k from params.k

  output:
  file 'masked_size.tsv' into masked_size_ch

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
  file 'masked_size' from masked_size_ch.collect()

  output:
  file 'masked_size.tsv' into masked_size_total_ch

  """
  #! /usr/bin/env python

  import glob
  import pandas as pd

  sum(pd.read_csv(file, sep='\t', index_col=0, header=None, squeeze=True)
      for file in glob.glob('masked_size*')).to_csv('masked_size.tsv', sep='\t', header=False)
  """
}

// mutation types for each chromosome vcf
process mutation_types {

  executor 'sge'
  memory '500 MB'
  time '1d'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'ancestor.fa', 'snps.vcf.gz', 'snps.vcf.gz.tbi' from mask_ch_2.join(ancestor_ch_2).join(vcf_ch)
  path 'integrated_call_samples.tsv' from params.samples
  val k from params.k

  output:
  tuple chrom, 'mutation_types.vcf.gz' into mutation_types_ch

  """
  # NOTE: sample NA18498 is missing in hg38 release of low coverage data, see https://www.biorxiv.org/content/10.1101/600254v1
  tail -n +2 integrated_call_samples.tsv | cut -f1 | grep -v NA18498 > all_samples.txt
  bcftools view -S all_samples.txt -c 1:minor -R mask.bed -m2 -M2 -v snps -f PASS -Ou snps.vcf.gz | bcftools view -g ^miss -Ou | mutyper variants ancestor.fa - --strict --k ${k} | bcftools convert -Oz -o mutation_types.vcf.gz
  """
}

populations_ch = Channel
    .fromPath(params.samples)
    .splitCsv(skip: 1, sep: '\t')
    .map{ row -> row[2] + '_' + row[1] }
    .unique()


process ksfs {

  executor 'sge'
  memory '500 MB'
  time '1d'
  // scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mutation_types.vcf.gz' from mutation_types_ch
  each pop from populations_ch
  path 'integrated_call_samples.tsv' from params.samples
  val k from params.k

  output:
  tuple pop, 'ksfs.tsv' into ksfs_ch

  """
  # NOTE: sample NA18498 is missing in hg38 release of low coverage data, see https://www.biorxiv.org/content/10.1101/600254v1
  grep ${pop.split('_')[1]} integrated_call_samples.tsv | cut -f1 | grep -v NA18498 > samples.txt
  bcftools view -S samples.txt -c 1:minor -G mutation_types.vcf.gz | mutyper ksfs - > ksfs.tsv
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
  tuple pop, 'ksfs' from ksfs_ch.groupTuple(size: chromosomes.size())

  output:
  tuple pop, 'ksfs.tsv' into ksfs_total_ch

  """
  #! /usr/bin/env python

  import glob
  import pandas as pd

  sum(pd.read_csv(file, sep='\t', index_col=0)
      for file in glob.glob('ksfs*')).to_csv('ksfs.tsv', sep='\t')
  """
}

ksfs_total_ch.into { ksfs_total_ch_1; ksfs_total_ch_2; ksfs_total_ch_3; ksfs_total_ch_4; ksfs_total_ch_5; ksfs_total_ch_6; ksfs_total_ch_7 }

alpha_tv = [0] + (0..4.5).by(0.5).collect { 10**it }
alpha_spline = [0] + (1..5.5).by(0.5).collect { 10**it }
alpha_ridge = 1e-4

beta_tv = 7e1
beta_spline = 0
beta_ridge = 1e-4
beta_rank = 0

ref_pop = 'False'
folded = 'False'

process eta_sweep {

  executor 'sge'
  memory '500 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/eta_sweep/${alpha_tv}_${alpha_spline}/${population}", mode: 'copy'

  input:
  tuple population, 'ksf.tsv' from ksfs_total_ch_1.filter { it[0] == 'EUR_CEU' }
  file 'masked_size.tsv' from masked_size_total_ch
  each alpha_tv from alpha_tv
  each alpha_spline from alpha_spline
  val alpha_ridge
  val beta_tv
  val beta_spline
  val beta_ridge
  val beta_rank
  val ref_pop
  val folded

  output:
  file 'dat.pkl' into eta_sweep_ch

  script:
  template 'infer.py'
}

alpha_tv = 1e2
alpha_spline = 3e3

beta_tv = [0] + (0..3).by(0.5).collect { 10**it }
beta_spline = [0] + (0..5.5).by(0.5).collect { 10**it }

process mu_sweep {

  executor 'sge'
  memory '500 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/mu_sweep/${beta_tv}_${beta_spline}/${population}", mode: 'copy'

  input:
  tuple population, 'ksf.tsv' from ksfs_total_ch_2.filter { it[0] == 'EUR_CEU' }
  file 'masked_size.tsv' from masked_size_total_ch
  val alpha_tv
  val alpha_spline
  val alpha_ridge
  each beta_tv from beta_tv
  each beta_spline from beta_spline
  each beta_ridge from beta_ridge
  val beta_rank
  val ref_pop
  val folded

  output:
  file 'dat.pkl' into mu_sweep_ch

  script:
  template 'infer.py'
}

alpha_tv = 1e2
alpha_spline = 3e3

beta_tv = 7e1
beta_spline = 0

process europulse {

  executor 'sge'
  memory '500 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/europulse/${population}", mode: 'copy'

  input:
  tuple population, 'ksf.tsv' from ksfs_total_ch_3.filter { it[0].split('_')[0] == 'EUR' }
  file 'masked_size.tsv' from masked_size_total_ch
  val alpha_tv
  val alpha_spline
  val alpha_ridge
  val beta_tv
  val beta_spline
  val beta_ridge
  val beta_rank
  val ref_pop
  val folded

  output:
  file 'dat.pkl' into europulse_ch

  script:
  template 'infer.py'
}

// same as above, but all populations and softer mutation spectrum history
alpha_tv = 1e2
alpha_spline = 3e3

beta_tv = 0
beta_spline = 5e4
beta_rank=1e2

process mush_ref {

  executor 'sge'
  memory '500 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/mush/${population}", mode: 'copy'

  input:
  tuple population, 'ksf.tsv' from ksfs_total_ch_4.first { it[0] == 'AFR_YRI' }
  file 'masked_size.tsv' from masked_size_total_ch
  val alpha_tv
  val alpha_spline
  val alpha_ridge
  val beta_tv
  val beta_spline
  val beta_ridge
  val beta_rank
  val ref_pop
  val folded

  output:
  file 'dat.pkl' into mush_ref_ch

  script:
  template 'infer.py'
}

alpha_ridge = 1e4
beta_ridge = 1e4
ref_pop = 'True'
process mush {

  executor 'sge'
  memory '500 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/mush/${population}", mode: 'copy'

  input:
  tuple population, 'ksf.tsv' from ksfs_total_ch_5.filter { it[0] != 'AFR_YRI' }
  file 'masked_size.tsv' from masked_size_total_ch
  val alpha_tv
  val alpha_spline
  val alpha_ridge
  val beta_tv
  val beta_spline
  val beta_ridge
  val beta_rank
  val ref_pop
  file 'dat.ref.pkl' from mush_ref_ch
  val folded

  output:
  file 'dat.pkl' into mush_ch

  script:
  template 'infer.py'
}

// same as previous two, but folded
alpha_ridge = 1e-4
beta_ridge = 1e-4
ref_pop = 'False'
folded = 'True'
process mush_ref_folded {

  executor 'sge'
  memory '500 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/mush_folded/${population}", mode: 'copy'

  input:
  tuple population, 'ksf.tsv' from ksfs_total_ch_6.first { it[0] == 'AFR_YRI' }
  file 'masked_size.tsv' from masked_size_total_ch
  val alpha_tv
  val alpha_spline
  val alpha_ridge
  val beta_tv
  val beta_spline
  val beta_ridge
  val beta_rank
  val ref_pop
  val folded

  output:
  file 'dat.pkl' into mush_ref_folded_ch

  script:
  template 'infer.py'
}

alpha_ridge = 1e4
beta_ridge = 1e4
ref_pop = 'True'
process mush_folded {

  executor 'sge'
  memory '500 MB'
  time '10m'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"
  publishDir "$params.outdir/mush_folded/${population}", mode: 'copy'

  input:
  tuple population, 'ksf.tsv' from ksfs_total_ch_7.filter { it[0] != 'AFR_YRI' }
  file 'masked_size.tsv' from masked_size_total_ch
  val alpha_tv
  val alpha_spline
  val alpha_ridge
  val beta_tv
  val beta_spline
  val beta_ridge
  val beta_rank
  val ref_pop
  file 'dat.ref.pkl' from mush_ref_folded_ch
  val folded

  output:
  file 'dat.pkl' into mush_folded_ch

  script:
  template 'infer.py'
}
