#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ancestor import Ancestor
from collections import Counter
from Bio.Seq import reverse_complement


def main():
    """
    usage: python masked_size.py -h
    """
    import argparse

    parser = argparse.ArgumentParser(description='print masked chromosome '
                                                 'size')
    parser.add_argument('anc_aln_file', type=str, default=None,
                        help='path to ancestral alignment fasta (one '
                             'chromosome)')
    parser.add_argument('mask_file', type=str, default=None,
                        help='path to bed file mask')
    parser.add_argument('--k', type=int, default=3,
                        help='kmer context')
    args = parser.parse_args()

    ancestor = Ancestor(args.anc_aln_file)
    nucs = set('ACGT')
    offset = args.k//2

    def line_gen():
        for line in open(args.mask_file, 'r'):
            chr, start, end = line.rstrip().split('\t')
            assert chr == ancestor.chr, f'bed/fasta chromosome mismatch: {chr} and {ancestor.chr}'
            yield start, end

    sizes = Counter(str(ancestor.anc_aln[(i - offset):(i + offset + 1)].seq)
                    for start, end in line_gen()
                    for i in range(max(int(start), offset),
                                   min(int(end), len(ancestor.anc_aln) - offset))
                    )

    # collapse reverse complements and remove ambiguous triplets
    for kmer in list(sizes.keys()):
        if not all(x in nucs for x in kmer):
            del sizes[kmer]
        elif kmer[offset] not in 'AC':
            sizes[reverse_complement(kmer)] += sizes[kmer]
            del sizes[kmer]

    assert len(sizes) == 2 * (args.k - 1) ** 4

    for kmer in sorted(sizes):
        print(f'{kmer}\t{sizes[kmer]}')


if __name__ == '__main__':
    main()
