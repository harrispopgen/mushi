#! /usr/bin/env python

import gzip


def main():
    """
    usage: python tobed.py -h
    """
    import argparse
    parser = argparse.ArgumentParser(description='create a file of genomic regions '
                                                 'based on excluding things like '
                                                 'conserved sites, repeats, etc. '
                                                 '0-based bed format')
    parser.add_argument('in_file',
                        type=str,
                        help='path to gzipped phastcons '
                             '(phastConsElements100way.txt.gz) or '
                             'repeatmastker file (nestedRepeats.txt.gz)')
    args = parser.parse_args()
    for line in gzip.open(args.in_file, 'r'):
        chr, start, end = line.decode('ascii').split('\t')[1:4]
        # remove the "chr" from e.g. "chr1"
        chr = chr[3:]
        print(f'{chr}\t{int(start) - 1}\t{end}')


if __name__ == '__main__':
    main()
