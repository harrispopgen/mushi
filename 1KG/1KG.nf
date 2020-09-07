#!/usr/bin/env nextflow

masks = file("/net/harris/vol1/data/phase3_1000genomes_supporting/accessible_genome_mask_hg38/pilot_strict_combined.allChr.mask.bed")
outgroup_fasta = file("/net/harris/vol1/data/panTro6/panTro6.fa")
chains = file("/net/harris/vol1/data/alignment_chains/hg38ToPanTro6.over.chain.gz")

params.hg38_dir = "/net/harris/vol1/data/hg38/"
vcf_dir = "/net/harris/vol1/nygc-transfered/"

chromosomes = 1..22

ref_fastagz_channel = Channel
    .from(chromosomes)
    .map { [it,
            file(params.hg38_dir + "chr${it}.fa.gz")]
          }

Channel
  .from(chromosomes)
  .map { [it,
          file(params.vcf_dir + "CCDG_13607_B01_GRM_WGS_2019-02-19_chr${it}.recalibrated_variants.vcf.gz")]
        }
  .into{ vcf_channel_1; vcf_channel_2 }

params.k = 3

process mask_and_ancestor {

  executor 'sge'
  memory '1 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  file masks
  tuple chrom, 'ref.fa.gz', 'snps.vcf.gz' from ref_fastagz_channel.join(vcf_channel_1)
  file outgroup_fasta
  file chains

  output:
  tuple chrom, 'mask.bed' into mask_channel
  tuple chrom, 'ancestor.fa' into ancestor_channel

  """
  zcat -f ref.fa.gz > ref.fa
  grep -P "^chr${chrom}*\\t.*strict\$" ${masks} | cut -f1-3 > mask.bed
  mutyper ancestor --bed mask.bed snps.vcf.gz ref.fa ${outgroup_fasta} ${chains} ancestor.fa
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
  tuple chrom, 'mask.bed', 'ancestor.fa' from mask_channel_1.join(ancestor_channel_1)
  val k from params.k

  output:
  tuple chrom, 'masked_size.tsv' into masked_size_channel

  """
  mutyper targets ancestor.fa --k ${k} --bed mask.bed > masked_size.tsv
  """
}

process mutation_types {

  executor 'sge'
  memory '5 GB'
  scratch true
  conda "${CONDA_PREFIX}/envs/1KG"

  input:
  tuple chrom, 'mask.bed', 'snps.vcf.gz', 'ancestor.fa' from mask_channel_2.join(vcf_channel_2).join(ancestor_channel_2)
  val k from params.k

  output:
  tuple chrom, 'mutation_types.bcf' into mutation_types_channel

  """
  bcftools view -T mask.bed -m2 -M2 -v snps -c 1:minor -Ou -f PASS -U snps.vcf.gz | mutyper variants ancestor.fa - --k ${k} | bcftools convert -Ob > mutation_types.bcf
  """
}



  //
  // # nested dict of superpopulation -> population -> sample list
  // superpops = defaultdict(lambda: defaultdict(list))
  // with open(samples) as f:
  //     f.readline()
  //     for line in f:
  //         sample, pop, superpop = line.split('\t')[:3]
  //         if pops is None or pop in pops:
  //             superpops[superpop][pop].append(sample)
  //
  // # loop over populations and compute sample frequency data
  // # for superpop in superpops:
  // #     for pop in superpops[superpop]:
  // #         # k-SFS
  // #         tgt = os.path.join(outdir, f'{k}-SFS.{superpop}.{pop}.tsv')
  // #         cmd = ('bcftools concat -n -Ou $SOURCES '
  // #                f'| bcftools view -s {",".join(superpops[superpop][pop])} '
  // #                '-c 1:minor -Ou '
  // #                '| mutyper ksfs - > $TARGET')
  // #         freqs = env.Command(tgt, bcf_mutation_types, cmd)
  //
  // # masked genome size for mutation rate estimation
  // tgt = os.path.join(outdir, f'masked_size.tsv')
  // cmd = 'python masked_size_aggregator.py $SOURCES > $TARGET'
  // masked_size = env.Command(tgt, masked_sizes, cmd)
