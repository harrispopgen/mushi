#!/usr/bin/env nextflow

params.outdir = "output"

params.n_change_points = 100
params.n_haplotypes = 200

process histories {

  executor 'sge'
  memory '1 GB'
  time '1m'
  // scratch true
  conda "${CONDA_PREFIX}/envs/simulation"
  publishDir "$params.outdir", pattern: '*.pdf', mode: 'copy'

  output:
  tuple 'eta.pkl', 'mu.pkl' into histories_ch
  tuple 'eta.pdf', 'mu.pdf' into histories_summary_ch

  """
  #!/usr/bin/env python

  import mushi
  import stdpopsim
  import numpy as np
  from matplotlib import pyplot as plt
  from scipy.special import expit
  import pickle

  # Time grid

  change_points = np.logspace(0, np.log10(100000), ${params.n_change_points})
  t = np.concatenate((np.array([0]), change_points))


  # Sawtooth demographic history eta(t)

  species = stdpopsim.get_species("HomSap")
  model = species.get_demographic_model("Zigzag_1S14")
  ddb = model.get_demography_debugger()
  eta = mushi.eta(change_points, 2 * ddb.population_size_trajectory(steps=t)[:, 0])

  plt.figure(figsize=(3.5, 3.5))
  eta.plot(c='k')
  plt.savefig('eta.pdf')

  pickle.dump(eta, open('eta.pkl', 'wb'))


  # Mutation rate history mu(t)
  # A 96 dimensional history with a mixture of two latent signature: constant and pulse.

  flat = np.ones_like(t)
  pulse = expit(.1 * (t - 100)) - expit(.01 * (t - 2000))
  ramp = expit(-.01 * (t - 100))
  n_mutation_types = 96
  Z = np.zeros((len(t), n_mutation_types))
  np.random.seed(0)

  Z[:, 0] = 1 * flat + .5 * pulse
  Z[:, 1] = 0.5 * flat + .4 * ramp

  for col in range(2, n_mutation_types):
      scale = np.random.lognormal(-.2, .3)
      pulse_weight = 5 if col == 0 else 0
      Z[:, col] = scale * flat

  mu = mushi.mu(change_points, Z)

  plt.figure(figsize=(4, 4))
  mu.plot(range(2, n_mutation_types), alpha=0.1, lw=2, c='C0', clr=True)
  mu.plot((0,), alpha=0.75, lw=3, c='C1', clr=True)
  mu.plot((1,), alpha=0.75, lw=3, c='C2', clr=True)
  plt.savefig('mu.pdf')

  pickle.dump(mu, open('mu.pkl', 'wb'))
  """
}

process ksfs {

  executor 'sge'
  memory '10 GB'
  time '1d'
  // scratch true
  conda "${CONDA_PREFIX}/envs/simulation"
  publishDir "$params.outdir", pattern: '{*.pdf,*.txt}', mode: 'copy'

  input:
  tuple 'eta.pkl', 'mu.pkl' from histories_ch

  output:
  tuple 'ksfs.pkl', 'eta.pkl', 'mu.pkl' into ksfs_ch
  tuple 'ksfs.pdf', 'n_trees.txt', 'n_variants.txt', 'tmrca_cdf.pdf', 'mu0.txt' into ksfs_summary_ch
  stdout ksfs_out

  """
  #!/usr/bin/env python

  import pickle
  import mushi
  import stdpopsim
  import msprime
  import numpy as np
  from matplotlib import pyplot as plt

  # Load histories
  eta = pickle.load(open('eta.pkl', 'rb'))
  mu = pickle.load(open('mu.pkl', 'rb'))

  # stdpopsim simulation
  # - n sampled haplotypes
  # - generate initial tree sequence without mutations, then we'll place mutations according to the MuSH

  n = ${params.n_haplotypes}
  chrom = 1
  species = stdpopsim.get_species('HomSap')
  contig = species.get_contig(f'chr{chrom}')
  contig = stdpopsim.Contig(recombination_map=contig.recombination_map, mutation_rate=0, genetic_map=contig.genetic_map)
  model = stdpopsim.PiecewiseConstantSize(eta.y[0] / 2, *zip(eta.change_points, eta.y[1:] / 2))
  samples = model.get_samples(n)
  engine = stdpopsim.get_default_engine()
  ts = engine.simulate(model, contig, samples, mutation_rate=0, seed=1)


  # Number of marginal trees
  print(ts.num_trees, file=open('n_trees.txt', 'w'))

  # Simulate k-SFS
  # - place mutations on simulated tree sequence according to mu(t)
  # - iterate over mutation types and epochs
  # - compute component of k-SFS for each iterate

  X = np.zeros((n - 1, mu.Z.shape[1]))
  for start_time, end_time, mutation_rate in mu.epochs():
      mutation_rate_total = mutation_rate.sum()
      print(f'epoch boundaries: ({start_time:.2f}, {end_time:.2f}), μ: {mutation_rate_total:.2f}     ',
            flush=True, end='     \\r')
      # note: the factor of 1 / chrom.length is needed to scale the mutation rate from sites to genomes
      ts_temp = msprime.mutate(ts,
                               rate=mutation_rate_total / species.genome.chromosomes[chrom - 1].length,
                               start_time=start_time,
                               end_time=end_time,
                               random_seed=1,
                               keep=True)
      for var in ts_temp.variants():
          i = var.genotypes.sum() - 1
          j = np.random.choice(mu.Z.shape[1], p=(mutation_rate / mutation_rate_total))
          X[i, j] += 1

  ksfs = mushi.kSFS(X=X)


  # Plot SFS and k-SFS

  plt.figure(figsize=(4, 3))
  ksfs.plot_total(kwargs=dict(ls='', marker='o', ms=5, c='k', alpha=0.75),
                  line_kwargs=dict(c='C0', alpha=0.75, lw=3),
                  fill_kwargs=dict(color='C0', alpha=0.1))
  plt.xscale('log')
  plt.yscale('log')

  plt.figure(figsize=(4, 3))
  ksfs.plot(range(2, mu.Z.shape[1]), clr=True, kwargs=dict(alpha=0.1, ls='', marker='.', c='C0'))
  ksfs.plot((0,), clr=True, kwargs=dict(alpha=0.75, ls='', marker='o', c='C1'))
  ksfs.plot((1,), clr=True, kwargs=dict(alpha=0.75, ls='', marker='o', c='C2'))
  plt.savefig('ksfs.pdf')

  pickle.dump(ksfs, open('ksfs.pkl', 'wb'))

  # Number of segregating sites
  print(ksfs.X.sum(), file=open('n_variants.txt', 'w'))

  # TMRCA CDF
  plt.figure(figsize=(3.5, 3.5))
  plt.plot(eta.change_points, ksfs.tmrca_cdf(eta))
  plt.xlabel('\$t\$')
  plt.ylabel('TMRCA CDF')
  plt.ylim([0, 1])
  plt.xscale('log')
  plt.tight_layout()
  plt.savefig('tmrca_cdf.pdf')

  # Estimate constant total mutation rate using most recent time point (introducing a misspecification)
  mu0 = mu.Z[0, :].sum()
  print(mu0, file=open('mu0.txt', 'w'))
  """
}


alpha_tv = [0] + (0..4.5).by(0.25).collect{ 10**it }
alpha_spline = [0] + (1..5.5).by(0.25).collect{ 10**it }
alpha_ridge = 1e-4

process infer_eta {

  executor 'sge'
  memory '1 GB'
  time '5m'
  scratch true
  conda "${CONDA_PREFIX}/envs/simulation"
  publishDir "$params.outdir/infer_eta/${alpha_tv}_${alpha_spline}", mode: 'copy'

  input:
  tuple 'ksfs.pkl', 'eta.pkl', 'mu.pkl' from ksfs_ch
  each alpha_tv from alpha_tv
  each alpha_spline from alpha_spline
  val alpha_ridge

  output:
  file 'dat.pkl' into infer_eta_ch
  file 'eta_fit.pdf' into infer_eta_summary_ch

  """
  #!/usr/bin/env python

  import pickle
  import mushi
  import matplotlib as mpl
  from matplotlib import pyplot as plt
  import numpy as np

  # Load ksfs and true histories
  ksfs = pickle.load(open('ksfs.pkl', 'rb'))
  eta_true = pickle.load(open('eta.pkl', 'rb'))
  mu_true = pickle.load(open('mu.pkl', 'rb'))

  # Estimate constant total mutation rate using most recent time point (introducing a misspecification)

  mu0 = mu_true.Z[0, :].sum()
  print(mu0, file=open('mu0.txt', 'w'))

  alpha_params = dict(alpha_tv=${alpha_tv}, alpha_spline=${alpha_spline}, alpha_ridge=${alpha_ridge})
  ksfs.infer_history(eta_true.change_points, mu0, **alpha_params,
                     tol=1e-16, max_iter=2000, infer_mu=False)


  # Plot results
  mpl.rc('text', usetex=True)
  mpl.rcParams['text.latex.preamble']=r"\\usepackage{amsmath}"

  plt.figure(figsize=(6, 3))
  plt.subplot(121)
  ksfs.plot_total(kwargs=dict(ls='', alpha=0.75, marker='o', ms=5, mfc='none', c='k', label=r'simulated SFS, \$\\mathbf x\$'),
                  line_kwargs=dict(c='r', ls=':', marker='.', ms=3, alpha=0.75, lw=1, label=r'reconstructed SFS, \$\\boldsymbol{\\xi}\$'),
                  fill_kwargs=dict(alpha=0))
  plt.xscale('log')
  plt.yscale('log')
  plt.legend(fontsize=8)

  plt.subplot(122)
  eta_true.plot(c='k', alpha=1, lw=3, label='true')
  ksfs.eta.plot(c='r', alpha=0.75, lw=2, label='inferred')
  plt.legend(fontsize=8, loc='upper right')
  plt.xlim([1e1, 5e4])
  plt.tight_layout()
  plt.savefig('eta_fit.pdf')

  pickle.dump([alpha_params['alpha_tv'], alpha_params['alpha_spline'], ksfs, eta_true], open('dat.pkl', 'wb'))
  """
}




// regularization_mu = dict(hard=False, beta_rank=2e1, beta_tv=1e2, beta_spline=1e3, beta_ridge=1e-4)
