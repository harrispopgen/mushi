#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cyvcf2
from collections import defaultdict, Counter
import pandas as pd


def main():
    """
    usage: python ksfs.py -h
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
        assert variant.INFO['AN'] == AN or AN is None, f'different AN {variant.INFO["AN"]} and {AN} indicate missing genotypes'
        AN = variant.INFO['AN']
        # biallelic snps only
        if (variant.is_snp and len(variant.ALT) == 1):
            ksfs_data[variant.INFO['mutation_type']][variant.INFO['AC']] += 1

    index = range(1, AN)
    for mutation_type in sorted(ksfs_data):
        ksfs_data[mutation_type] = [ksfs_data[mutation_type][ac] for ac in index]
    ksfs = pd.DataFrame(ksfs_data, index).reindex(sorted(ksfs_data), axis='columns')
    ksfs.to_csv(sys.stdout, sep='\t')


if __name__ == '__main__':
    main()
