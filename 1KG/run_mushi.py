#!/usr/bin/env python
# coding: utf-8

# Run mushi on k-SFS file

from .. import mushi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

def main():
    """
    usage: python run_mushi.py -h
    """
    import argparse

    parser = argparse.ArgumentParser(description='write snps with kmer context'
                                                 ' to stdout')
    parser.add_argument('ksfs', type=str, default=None,
                        help='path to k-SFS file')
    parser.add_argument('size', type=str, default=None,
                        help='path to file with masked genome size')
    parser.add_argument('mutation_rate', type=np.float, default=None,
                        help='site-wise mutation rate')
    parser.add_argument('outbase', type=str, default=None,
                        help='base name for output files')

    args = parser.parse_args()

    # plt.style.use('dark_background')

    # load k-SFS
    ksfs_df = pd.read_csv(args.ksfs, sep='\t', index_col=0)
    mutation_types = ksfs_df.columns
    n = ksfs_df.shape[0] + 1
    ksfs = mushi.kSFS(X=ksfs_df.values, mutation_types=mutation_types)

    # rank plot of the number of SNPs of each triplet mutation type
    plt.figure(figsize=(15, 3))
    plt.plot(ksfs_df.sum().sort_values(ascending=False).to_frame(), '.')
    plt.xticks(rotation='vertical', family='monospace')
    plt.ylabel('number of SNPs')
    plt.yscale('symlog')
    plt.tight_layout()
    plt.savefig(f'{args.outbase}.rank_snp_types.pdf')

    # genome size and mutation rate estimation
    masked_size = int(open(args.masked_size).read())
    μ_0 = args.mutation_rate * masked_size
    print(f'mutation rate in units of mutations per masked genome per generation: {μ_0}')

    # mutation type enrichment as a heatmap with correlation clustering
    ksfs.clustermap(figsize=(25, 10))
    plt.savefig(f'{args.outbase}.1KG_heatmap.pdf', transparent=True)

    # Initialize to constant
    change_points = np.logspace(0, 5.3, 200)

    # mask sites
    mask = np.array([False if (0 <= i <= n - 20) else True for i in range(n - 1)])

    ksfs.infer_constant(change_points=change_points, μ_0=μ_0, mask=mask)

    f_trajectory = []

    sweeps = 10
    tol = 1e-10
    f_old = None
    for sweep in range(1, 1 + sweeps):
        print(f'block coordinate descent sweep {sweep:.2g}')
        f = ksfs.coord_desc(# loss function parameters
                            fit='prf',
                            mask=mask,
                            # η(t) regularization parameters
                            α_tv=0,#1e3,
                            α_spline=1e3,
                            α_ridge=1e-10,
                            # μ(t) regularization parameters
                            β_rank=3e3,
                            β_tv=0,
                            β_spline=2e5,
                            β_ridge=1e-10,
                            hard=True,
                            # convergence parameters
                            max_iter=10000,
                            tol=tol,
                            γ=0.8)
        print(f'cost: {f}')
        if sweep > 1:
            relative_change = np.abs((f - f_old) / f_old)
            print(f'relative change: {relative_change:.2g}')
        print()
        f_old = f
        f_trajectory.append(f)

        if sweep > 1 and relative_change < tol:
            break

    plt.figure(figsize=(4, 2))
    plt.plot(f_trajectory)
    plt.xlabel('iterations')
    plt.ylabel('cost')
    # plt.xscale('symlog')
    plt.tight_layout()
    plt.savefig(f'{args.outbase}.iterations.pdf')

    plt.figure(figsize=(6, 6))
    plt.subplot(221)
    ksfs.plot_total()
    plt.subplot(222)
    ksfs.η.plot()
    plt.subplot(223)
    ksfs.plot(normed=True, alpha=0.5)
    plt.subplot(224)
    ksfs.μ.plot(normed=True, alpha=0.5)
    plt.savefig(f'{args.outbase}.fit.pdf')

    # pickle the final ksfs (which contains all the inferred history info)
    pickle.dump(ksfs, f'{args.outbase}.pkl')


if __name__ == '__main__':
    main()
