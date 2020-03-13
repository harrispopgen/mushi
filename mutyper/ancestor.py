#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cyvcf2
import pyfaidx
import re
from Bio.Seq import reverse_complement


class Ancestor():
    nts = ('A', 'C', 'G', 'T')

    def __init__(self, anc_fasta_file: str, k: int = 3, target: int = None,
                 strand_file: str = None):
        """ancestral state of a chromosome

        anc_fasta_file: path to ancestral sequence fasta
        k: the size of the context window (default 3)
        target: which position for the site within the kmer (default middle)
        strand_file: path to bed file with regions where reverse strand defines
                     mutation context, e.g. direction of replication or
                     transcription (default collapse reverse complements)
        """
        self.fasta = pyfaidx.Fasta(anc_fasta_file, read_ahead=10000,
                                   as_raw=True)
        if target is None:
            assert k % 2 == 1, f'k = {k} must be odd for default middle target'
            target = k // 2
        assert 0 <= target < k
        self.target = target
        self.k = k
        if strand_file is None:
            self.strandedness = None
            assert self.target == self.k // 2, f'non-central {self.target} target requires strand_file'
        else:
            raise NotImplementedError('strand_file argument must be None')

    def stranded_context(self, id: str, pos: int) -> str:
        """ancestral context of sequence id and pos, oriented or collapsed by
        strand (returns None if ancestral state at target not in capital ACGT)

        id: fasta record identifier
        pos: position (0-based)
        """
        start = pos - self.target
        assert start >= 0
        end = pos + self.k - self.target
        assert start <= end
        context = self.fasta[id][start:end]
        if self.strandedness is None:
            if context[self.target] in 'AC':
                return context
            elif context[self.target] in 'TG':
                return reverse_complement(context)
            else:
                return None
        else:
            raise NotImplementedError('self.strandedness must be None')

    def mutation_type(self, id: str, pos: int, ref: str, alt: str) -> str:
        """mutation type of a given snp, oriented or collapsed by strand
        returns a tuple of ancestral and derived kmers, or None

        id: fasta record identifier
        pos: position (0-based)
        ref: reference allele (A, C, G, or T)
        alt: alternative allele (A, C, G, or T)
        """
        # ancestral state
        anc = self.fasta[id][pos]
        # derived state
        if anc == ref:
            der = alt
        elif anc == alt:
            der = ref
        else:
            der = '?'
        start = pos - self.target
        assert start >= 0
        end = pos + self.k - self.target
        assert start <= end

        # NOTE: don't wanted stranded_context method here
        context = self.fasta[id][start:end]
        anc_kmer = f'{context[:self.target]}{anc}{context[(self.target + 1):]}'
        der_kmer = f'{context[:self.target]}{der}{context[(self.target + 1):]}'


        if self.strandedness is None:
            if anc in 'AC':
                return anc_kmer, der_kmer
            elif anc in 'TG':
                return reverse_complement(anc_kmer), reverse_complement(der_kmer)
            else:
                return None, None

def main():
    """
    usage: python ancestor.py -h
    """
    import argparse

    parser = argparse.ArgumentParser(description='add kmer context to vcf/bcf'
                                                 'INFO and stream to stdout')
    parser.add_argument('anc_fasta_file', type=str, default=None,
                        help='path to ancestral alignment fasta (one '
                             'chromosome)')
    parser.add_argument('--vcf_file', type=str, default='-',
                        help='VCF file (default stdin)')
    parser.add_argument('--k', type=int, default=3,
                        help='k-mer context size (default 3)')
    parser.add_argument('--target', type=int, default=None,
                        help='0-based mutation target position in kmer (default middle)')
    parser.add_argument('--sep', type=str, default=':',
                        help='field delimiter in fasta headers (default ":")')
    parser.add_argument('--chrom_pos', type=int, default=2,
                        help='0-based chromosome field position in fasta headers (default 2)')
    parser.add_argument('--strand_file', type=str, default=None,
                        help='path to bed file with regions where reverse '
                             'strand defines mutation context, e.g. direction '
                             'of replication or transcription (default collapse '
                             'reverse complements)')


    args = parser.parse_args()

    ancestor = Ancestor(args.anc_fasta_file, args.k, args.target, args.strand_file)

    # parse chromosome names from fasta headers
    chrom_map = {name.split(args.sep)[args.chrom_pos]:
                 name for name in ancestor.fasta.keys()}

    vcf = cyvcf2.VCF(args.vcf_file)
    vcf.add_info_to_header({'ID': 'mutation_type',
                            'Description': f'ancestral {args.k}mer mutation type',
                            'Type': 'Character', 'Number': '1'})
    vcf_writer = cyvcf2.Writer('-', vcf)
    vcf_writer.write_header()
    for variant in vcf:
        # biallelic snps only
        if not (variant.is_snp and len(variant.ALT) == 1):
            continue
        # mutation type
        anc_kmer, der_kerm = ancestor.mutation_type(chrom_map[variant.CHROM],
                                                    variant.start, variant.REF,
                                                    variant.ALT[0])
        mutation_type = f'{anc_kmer}>{der_kerm}'
        # keep snps with ancestral states in kmer context (indicated by
        # capital ACGT for high confidence, lowercase for low confidence)
        if re.match('^[ACGT]+>[ACGT]+$', mutation_type) is None:
            continue
        variant.INFO['mutation_type'] = mutation_type
        vcf_writer.write_record(variant)


if __name__ == '__main__':
    main()
