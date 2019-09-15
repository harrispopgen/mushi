#! /usr/bin/env python
# -*- coding: utf-8 -*-

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
import re, sys
import pandas as pd

class AncestralStates():
    def __init__(self, aln: SeqRecord, outgroup_aln: SeqRecord=None, df_seg: pd.DataFrame=None,
                 verbose=False):
        """
        ancestral states of a chromosome
        two constructors: load ancestral aln directly, or compute from reference, outgroup, and snps
        @param aln: single record for a chr of ancestral sequence OR reference genome, e.g. hg19.1.fa if the other args are provided
        @param outgroup_aln: outgroup (i.e. chimp) Bio.SeqRecord alignment for same chr
                                    (single record, e.g. hg19ancNoChr.1.fa)
        @param df_seg: pandas dataframe of snps indexed by (chr, pos) with columns [chr, pos, ref, alt, ac, an]
        """
        self.verbose = verbose
        fields = aln.id.split(':')
        self.chr = int(fields[0])
        if len(fields) > 1:
            self.start = int(fields[1].split('-')[0])
        else:
            self.start = 1
        if outgroup_aln is None and df_seg is None:
            # assert set(self.ancestral_alignment) == set('ACGT')
            self.ancestral_alignment = aln.upper()
        else:
            assert aln.id == outgroup_aln.id
            # initalize ascertainment to the reference, all uppercase
            ascertained = aln.seq.upper().tomutable()
            # iterate over seg sites and possibly change to chimp or fixed nonref
            n_seg_sites = df_seg.shape[0]
            for ct, (chr2, site) in enumerate(df_seg.index, 1):
                if self.verbose:
                    print('computing ancestral states for chr {} pos {}\r'.format(chr2, site), end='', flush=True)
                assert chr2 == self.chr
                ref_char = aln.seq[site - self.start].upper()
                anc_char = outgroup_aln.seq[site - self.start].upper()
                assert ref_char == df_seg.ref[(self.chr, site)]
                assert anc_char in 'ACGTN'
                # if the site is segregating, use the outgroup for ancestral state ascertainment
                # fixed
                if df_seg.ac[(self.chr, site)] == df_seg.an[(self.chr, site)]:
                    ascertained[site - self.start] = df_seg.alt[(self.chr, site)]
                elif df_seg.ac[(self.chr, site)] == 0:
                    ascertained[site - self.start] = df_seg.ref[(self.chr, site)]
                # segregating
                else:
                    if df_seg.ref[(self.chr, site)] == anc_char or df_seg.alt[(self.chr, site)] == anc_char:
                        ascertained[site - self.start] = anc_char
                    else: # not biallelic NOTE should I just take the major allele instead?
                        ascertained[site - self.start] = 'N'
            if self.verbose:
                print('')

            self.ancestral_alignment = SeqRecord(ascertained, id=aln.id, description='')

    def get_state(self, pos: int) -> str:
        '''return ancestral state at site pos (1-based)'''
        return self.ancestral_alignment.seq[pos - self.start]

    def write(self, file=sys.stdout):
        '''print fasta record to file (default stdout)'''
        SeqIO.write(self.ancestral_alignment, file, 'fasta')


def main():
    """
    print ancestral alignment to stdout
    usage: python ancestral_states.py -h
    """
    import argparse
    parser = argparse.ArgumentParser(description='infer ancestral alignment for'
                                     ' a population using chimp alignment')
    parser.add_argument('reference_alignment',
                        type=str,
                        help='path to fasta reference genome alignment '
                             '(e.g. hg19.fa)')
    parser.add_argument('outgroup_alignment',
                        type=str,
                        help='path to fasta alignment of ancestral sequence '
                             '(e.g. hg19ancNoChr.fa)')
    parser.add_argument('segregating_sites',
                        type=str,
                        help='path to refpop.snps.tsv file (see SConstruct for format)')

    args = parser.parse_args()

    # parse fastas to Bio.SeqRecords
    ref_aln = SeqIO.read(args.reference_alignment, 'fasta')
    outgroup_aln = SeqIO.read(args.outgroup_alignment, 'fasta')

    # parse segregating sites into data frame
    df_seg = pd.read_csv(args.segregating_sites, sep='\t', header=None,
                         names=('chr', 'pos', 'ref', 'alt', 'ac', 'an'))
    df_seg = df_seg.set_index(['chr', 'pos'])
    # weirdly, bcftools view can sometimes produce repeated output positions,
    # despite passing the -m2 and -M2 options. We exclude these.
    df_seg = df_seg[~df_seg.index.duplicated(keep=False)]

    ancestral_states = AncestralStates(ref_aln, outgroup_aln, df_seg)
    ancestral_states.write()


if __name__ == '__main__':
    main()
