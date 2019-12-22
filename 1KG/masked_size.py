#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ancestor import Ancestor


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

    size = 0
    for line in open(args.mask_file, 'r'):
        chr, start, end = line.rstrip().split('\t')
        assert chr == ancestor.chr
        for i in range(max(int(start), offset),
                       min(int(end), len(ancestor.anc_aln) - offset)):
            kmer = ancestor.anc_aln[(i - offset):(i + offset + 1)]
            if all(x in nucs for x in kmer):
                size += 1
    print(size)


if __name__ == '__main__':
    main()
