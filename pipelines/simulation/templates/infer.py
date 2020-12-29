#!/usr/bin/env python

import pickle
import mushi
import numpy as np

# Load ksfs and true histories
ksfs = pickle.load(open('ksfs.pkl', 'rb'))
mu0 = pickle.load(open('mu0.pkl', 'rb'))

change_points = np.logspace(0, np.log10(100000), ${params.n_change_points})

alpha_params = dict(alpha_tv=${alpha_tv}, alpha_spline=${alpha_spline}, alpha_ridge=${alpha_ridge})
beta_params = dict(beta_tv=${beta_tv}, beta_spline=${beta_spline}, beta_ridge=${beta_ridge})
convergence_params = dict(tol=1e-16, max_iter=2000)

if ${freq_mask}:
    clip_low = 0
    clip_high = 20
    freq_mask = np.array([True if (clip_low <= i < ksfs.n - clip_high - 1) else False
                          for i in range(ksfs.n - 1)])
else:
    freq_mask = None

ksfs.infer_history(change_points, mu0, loss='prf',
                   infer_mu=False, folded=${folded},
                   mask=(None if ${folded} else freq_mask),
                   **alpha_params, **convergence_params)
ksfs.infer_history(change_points, mu0, loss='prf',
                   infer_eta=False,
                   mask=freq_mask,
                   **beta_params, **convergence_params)

pickle.dump([alpha_params, beta_params, ksfs], open('dat.pkl', 'wb'))
