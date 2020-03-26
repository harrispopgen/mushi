#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cyvcf2
from collections import defaultdict, Counter
import pandas as pd


def main():
    """
    usage: python ksfs.py -h

    prints the k-SFS, or the mutation spectrum if input contains only one sample
    """
    import argparse

    parser = argparse.ArgumentParser(description='compute ksfs from vcf with '
                                                 'INFO/mutation_type data')
    parser.add_argument('--vcf_file', type=str, default='-',
                        help='VCF file (default stdin)')

    args = parser.parse_args()

    vcf = cyvcf2.VCF(args.vcf_file)

    ksfs_data = defaultdict(lambda: Counter())
    AN = None
    for variant in vcf:
        # AN must be the same for all sites (no missing genotypes)
        if AN is not None and variant.INFO['AN'] != AN:
            raise ValueError(f'different AN {variant.INFO["AN"]} and {AN} indicates missing genotypes')
        AN = variant.INFO['AN']
        ksfs_data[variant.INFO['mutation_type']][variant.INFO['AC']] += 1

    index = range(1, AN)
    for mutation_type in sorted(ksfs_data):
        ksfs_data[mutation_type] = [ksfs_data[mutation_type][ac] for ac in index]
    ksfs = pd.DataFrame(ksfs_data, index).reindex(sorted(ksfs_data), axis='columns')
    try:
        print(ksfs.to_csv(sep='\t', index=True if ksfs.shape[0] > 1 else False,
              index_label='sample_frequency'))
    except BrokenPipeError:
        pass

if __name__ == '__main__':
    main()
