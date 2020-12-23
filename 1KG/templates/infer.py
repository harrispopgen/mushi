#!/usr/bin/env python

import pickle
import mushi
import pandas as pd
import numpy as np

ksfs = mushi.kSFS(file='ksf.tsv')

# reference eta for ancestral fusion
if ${ref_pop}:
    ref_pop = pickle.load(open('dat.ref.pkl', 'rb'))[2]
    eta_ref = ref_pop.eta
    mu_ref = ref_pop.mu
else:
    eta_ref = None
    mu_ref = None

# sorts the columns of the ksfs
sorted_triplets = [f'{a5}{a}{a3}>{a5}{d}{a3}' for a in 'AC'
                   for d in 'ACGT' if d != a
                   for a5 in 'ACGT' for a3 in 'ACGT']
foo, bar = ksfs.mutation_types.reindex(sorted_triplets)
ksfs.mutation_types = foo
ksfs.X = ksfs.X[:, bar]

masked_genome_size = pd.read_csv('masked_size.tsv', sep='\t', header=None, index_col=0, names=('count',))

change_points = np.logspace(np.log10(1), np.log10(200000), 200)

u = 1.25e-8
mu0 = u * masked_genome_size['count'].sum()

if not ${folded}:
    clip_low = 0
    clip_high = 10
    freq_mask = np.array([True if (clip_low <= i < ksfs.n - clip_high - 1) else False
                          for i in range(ksfs.n - 1)])
else:
    freq_mask = None

convergence_params = dict(tol=1e-16, max_iter=2000)
alpha_params = dict(alpha_tv=${alpha_tv}, alpha_spline=${alpha_spline}, alpha_ridge=${alpha_ridge})
beta_params = dict(beta_tv=${beta_tv}, beta_spline=${beta_spline}, beta_ridge=${beta_ridge}, beta_rank=${beta_rank})
ksfs.infer_history(change_points, mu0, loss='prf', mask=freq_mask,
                   eta_ref=eta_ref, infer_mu=False, folded=${folded},
                   **alpha_params, **convergence_params)
ksfs.infer_history(change_points, mu0, loss='prf', mask=freq_mask,
                   mu_ref=mu_ref, infer_eta=False,
                   **beta_params, **convergence_params)

pickle.dump([alpha_params, beta_params, ksfs, '${population}'], open('dat.pkl', 'wb'))
