#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gzip


def main():
    """
    usage: python region_mask.py -h
    """
    import argparse
    parser = argparse.ArgumentParser(description='create a file of genomic regions '
                                                 'based on excluding things like '
                                                 'conserved sites, repeats, etc. '
                                                 '1-based tab delimited format for '
                                                 'the -T option in bcftools: '
                                                 'https://samtools.github.io/bcftools/bcftools.html')
    parser.add_argument('phastcons_file',
                        type=str,
                        help='path to gzipped phastcons file phastConsElements100way.txt.gz, '
                             'downloaded from UCSC Genome Browser')
    parser.add_argument('repeatmasker_file',
                        type=str,
                        help='path to gzipped repeatmasker file nestedRepeats.txt.gz, '
                             'downloaded from UCSC Genome Browser')
    args = parser.parse_args()

    # The idea is to create  list of regions that we'll want to exclude with
    # something like `bcftools view -T ^foo.tsv.gz...`

    for file in (args.phastcons_file, args.repeatmasker_file):
        for line in gzip.open(file, 'r'):
            chr, start, end = line.decode('ascii').split('\t')[1:4]
            # remove the "chr" from e.g. "chr1"
            chr = chr[3:]
            print('\t'.join((chr, start, end)))


if __name__ == '__main__':
    main()
