#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ancestor import Ancestor
from collections import Counter


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

    args = parser.parse_args()

    ancestor = Ancestor(args.anc_aln_file)
    counts = Counter(ancestor.anc_aln)

    for line in open(args.mask_file, 'r'):
        chr, start, end = line.rstrip().split('\t')
        assert chr == ancestor.chr
        for char in ancestor.anc_aln[int(start):int(end)]:
            counts[char] -= 1
    print(counts['A'] + counts['C'] + counts['G'] + counts['T'])


if __name__ == '__main__':
    main()
