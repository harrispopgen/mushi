#!/usr/bin/env python

import pickle
import mushi

# Load ksfs and true histories
ksfs = pickle.load(open('ksfs.pkl', 'rb'))
eta_true = pickle.load(open('eta.pkl', 'rb'))
mu_true = pickle.load(open('mu.pkl', 'rb'))

# Estimate constant total mutation rate using most recent time point (introducing a misspecification)

mu0 = mu_true.Z[0, :].sum()
print(mu0, file=open('mu0.txt', 'w'))

alpha_params = dict(alpha_tv=${alpha_tv}, alpha_spline=${alpha_spline}, alpha_ridge=${alpha_ridge})
beta_params = dict(beta_tv=${beta_tv}, beta_spline=${beta_spline}, beta_ridge=${beta_ridge})
ksfs.infer_history(eta_true.change_points, mu0,
                 **alpha_params, **beta_params,
                 tol=1e-16, max_iter=2000)

pickle.dump([alpha_params, beta_params, ksfs, eta_true, mu_true], open('dat.pkl', 'wb'))
