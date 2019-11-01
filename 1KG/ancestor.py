#! /usr/bin/env python
# -*- coding: utf-8 -*-

from Bio import SeqIO
from Bio.Seq import reverse_complement
from typing import Tuple
import numpy as np
import re
import sys
import pandas as pd
from pandas.api.types import CategoricalDtype


class Ancestor():
    nts = ('A', 'C', 'G', 'T')

    def __init__(self, anc_aln_file: str):
        '''ancestral state of a chromosome

        anc_aln_file: path to ancestral sequence fasta
        '''

        # parse fasta to Bio.SeqRecord
        self.anc_aln = SeqIO.read(anc_aln_file, 'fasta')
        self.chr = self.anc_aln.id.split(':')[2]
        if int(self.chr) not in range(1, 23):
            raise ValueError(f'chromosome {self.chr} not in 1-22')
        self.anc_aln_len = len(self.anc_aln)

    def get_state(self, pos: int) -> str:
        '''return ancestral state at site pos (1-based)'''
        if pos > self.anc_aln_len:
            print(f'position {pos} is beyond chromosome length '
                  f'{self.anc_aln_len}')
            return None
        return self.anc_aln.seq[pos - 1]

    def write(self, file=sys.stdout):
        '''print fasta record to file (default stdout)'''
        SeqIO.write(self.anc_aln, file, 'fasta')

    def motif_sites(self, motif: str = 'TCC', target: int = None) -> pd.Index:
        '''return pandas frame of all sites that have a given motif context

        motif: sequence, can be regex, (default "TCC")
        target: which position of the site within the motif (default middle)
        '''
        if target is None:
            assert len(motif) % 2 == 1
            target = len(motif) // 2
        else:
            target = target
        if motif[target] not in 'ACGT':
            raise ValueError(f'motif {motif} at target position {target} must'
                             ' be a nucleotide')
        # add the reverse complement of the motif to make a regular expression
        motif_complement = reverse_complement(motif)
        motif_regex = re.compile(motif + '|' + motif_complement)
        sites = []
        for match in motif_regex.finditer(str(self.anc_aln.seq)):
            # if it's revcomp match, adjust target accordingly
            if match.group(0) == motif:
                this_target = target
            elif match.group(0) == motif_complement:
                this_target = len(motif) - target - 1
            else:
                raise
            # NOTE: ANGSD uses 1-based indexing
            sites.append((self.chr,
                          match.start(0) + this_target + 1))
        return pd.Index(sites)

    def site_motif(self, pos: int, der: str, k: int = 3,
                   target: int = None) -> str:
        '''return context of a given site

        pos: integer position NOTE: 1-based
        der: derived allele (A, C, G, or T)
        k: the size of the context window
        target: which position of the site within the motif (default middle)
        '''
        # this only makes sense if middle base is targeted
        assert k % 2 == 1
        target = k // 2

        focal_site_idx = pos - 1
        start = focal_site_idx - target
        assert start >= 0
        end = focal_site_idx + k - target
        assert end <= len(self.anc_aln.seq)
        assert start <= end

        # return the context or revcomp, whichever has an A or C at the target
        # base
        motif = str(self.anc_aln.seq[start:end])
        if motif[target] not in 'AC':
            motif = reverse_complement(motif)
            der = reverse_complement(der)
            target = k - target - 1
        return motif + '>' + motif[:target] + der + motif[(target + 1):]

    @staticmethod
    def kmers(k: int) -> Tuple[str]:
        '''all kmers of length k'''
        if k == 1:
            return Ancestor.nts
        else:
            return tuple(prefix + nt for prefix in Ancestor.kmers(k - 1)
                         for nt in Ancestor.nts)

    def context(self, snps_file: str, k: int = 3) -> pd.DataFrame:
        '''return snps with k-mer context info

        snps_file: path to SNPs file with columns
                   CHROM POS REF ALT AC AN
        k: kmer context size (default 3)
        '''
        # categorical dtype for nucleotides
        nuc_dtype = CategoricalDtype(list('ACGTN'))
        snps = pd.read_csv(snps_file, sep='\t', header=None,
                           names=('chr', 'pos', 'ref', 'alt', 'ac', 'an'),
                           dtype={'chr': np.uint8, 'pos': np.uint32,
                                  'ref': nuc_dtype, 'alt': nuc_dtype,
                                  'ac': np.uint16, 'an': np.uint16},
                           na_filter=False, engine='c').set_index(['chr',
                                                                   'pos'])
        # Weirdly, `bcftools view` can sometimes produce repeated output
        # positions, despite passing the `-m2` and `-M2` options. We exclude
        # these.
        snps = snps.loc[~snps.index.duplicated(keep=False)]

        # number of haplotypes
        n = snps.an[0]
        assert snps.an.unique() == [n]
        snps.rename(columns={'an': 'n'}, inplace=True)

        # update ac column name
        snps.rename(columns={'ac': 'sample frequency'}, inplace=True)

        # exclude fixed sites
        snps = snps[(0 < snps['sample frequency'])
                    & (snps['sample frequency'] < n)]

        # ensure that snps are from same chromosome as self
        assert all(str(chro) == self.chr for chro, pos in snps.index)

        snps['ancestral'] = [self.get_state(pos) for _, pos in snps.index]
        # keep sites that are biallelic and have defined ancestral state
        keep = ((snps.ancestral == snps.ref) | (snps.ancestral == snps.alt))
        snps = snps.loc[keep, :]

        # derived state
        snps.loc[snps.ancestral == snps.ref, 'derived'] = snps.alt
        snps.loc[snps.ancestral == snps.alt, 'derived'] = snps.ref

        # drop alt/ref columns now that we have ancestral/derived
        snps.drop(['ref', 'alt'], axis=1, inplace=True)

        # local nucleotide context of each snp
        snps['mutation type'] = [self.site_motif(pos,
                                                 snps.derived[(snps_chr, pos)],
                                                 k=k)
                                 for snps_chr, pos in snps.index]

        # drop ancestral and derived columns, now that we have mutation type
        snps.drop(['ancestral', 'derived'], axis=1, inplace=True)

        # keep snps with ancestral states in kmer context (indicated by
        # capital ACGT for high confidence, lowercase for low confidence)
        snps = snps[snps['mutation type'].str.match('^[ACGT]+>[ACGT]+$')]
        snps['mutation type'] = snps['mutation type'].str.upper()

        return snps


def main():
    """
    usage: python ancestor.py -h
    """
    import argparse

    parser = argparse.ArgumentParser(description='write snps with kmer context'
                                                 ' to stdout')
    parser.add_argument('anc_aln_file', type=str, default=None,
                        help='path to ancestral alignment fasta (one '
                             'chromosome)')
    parser.add_argument('snps_file', type=str, default=None,
                        help='path to SNPs file with columns '
                             'CHROM POS REF ALT AC AN')
    parser.add_argument('k', type=int, default=3,
                        help='k-mer context size (default 3)')

    args = parser.parse_args()

    ancestor = Ancestor(args.anc_aln_file)
    snps_context = ancestor.context(args.snps_file, k=args.k)
    snps_context.to_csv(sys.stdout, sep='\t')


if __name__ == '__main__':
    main()
