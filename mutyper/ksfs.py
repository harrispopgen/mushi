#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import sys


def main():
    """
    usage: python ksfs.py -h
    """
    import argparse

    parser = argparse.ArgumentParser(description='compute ksfs from snps.kmer '
                                                 'files')
    parser.add_argument('--variants_file', type=str, default=sys.stdin,
                        help='variant tsv file with columns CHRO, POS, REF, ALT, AC, AN, mutation_type')

    args = parser.parse_args()

    variants = pd.read_csv(args.variants_file, sep='\t',
                       names=('CHRO', 'POS', 'REF', 'ALT', 'AC', 'AN', 'mutation type'))
    n = variants.AN.iloc[0]
    assert all(variants.AN == n), f'variant file {args.variants_file} contains inconsistent n'
    ksfs = variants.groupby(['AC', 'mutation type']).size().unstack('mutation type',
                                                                    fill_value=0)

    ksfs.to_csv(sys.stdout, sep='\t')


if __name__ == '__main__':
    main()
