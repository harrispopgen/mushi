#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Projection analysis for specific triplet mutation spectra"""

from __future__ import print_function
import os
import glob
# import sconsutils
from SCons.Script import Environment, AddOption, GetOption

# this is where we'd define qsub commands
# sconsutils

# Set up SCons environment
environ = os.environ.copy()
env = Environment(ENV=environ)

# command line arguments
AddOption('--ref',
          dest='ref',
          type='string',
          metavar='<path>',
          help='path to directory containing bcf file(s) for reference '
               'population')
ref_bcf_dir = GetOption('ref')
AddOption('--sample_pops',
          dest='sample_pops',
          type='string',
          metavar='<path>',
          default=None,
          help='optional path to file populations for each sample (i.e. '
               'integrated_call_samples_v3.20130502.ALL.panel)')
sample_pops = GetOption('sample_pops')
AddOption('--pop',
          dest='pop',
          type='string',
          metavar='<path>',
          default=None,
          help='optional population to restrict panel (e.g. CEU or EUR)')
pop = GetOption('pop')
AddOption('--hg19',
          dest='hg19',
          type='string',
          metavar='<path>',
          help='path to hg19 fasta (should be indexed)')
hg19 = GetOption('hg19')
AddOption('--hg19_ancestral',
          dest='hg19_ancestral',
          type='string',
          metavar='<path>',
          help='path to hg19 chimp alignment fasta (should be indexed)')
hg19_ancestral = GetOption('hg19_ancestral')
AddOption('--phastcons',
          dest='phastcons',
          type='string',
          metavar='<path>',
          help='path to gzipped phastcons file (i.e. phastCons100way.txt.gz)')
phastcons = GetOption('phastcons')
AddOption('--repeats',
          dest='repeats',
          type='string',
          metavar='<path>',
          help='path to gzipped repeats file (i.e. nestedRepeats.txt.gz)')
repeats = GetOption('repeats')
AddOption('--debug_mode',
          dest='debug_mode',
          action='store_true',
          metavar='<debug_mode>',
          help='debug_mode: run on only a subset of each chr for speed')
debug = GetOption('debug_mode')
debug_range = ':30000000-31000000'
AddOption('--outdir',
          type='string',
          metavar="DIR",
          default='scons_output',
          help='directory in which to output results will be created if '
               'nonexistant (default "scons_output")')
outdir = GetOption('outdir')

# parse the --jobs command line argument
n_jobs = GetOption('num_jobs')

# need this so scons -h runs without error
if ref_bcf_dir is not None:
    # if we specified a population for the reerence snp panel, make a list of
    # those indvs
    if pop is not None:
        pop_list = env.Command([os.path.join(outdir, pop + '.list')],
                               [pop, sample_pops],
                               'grep ${SOURCES} | cut -f1 > ${TARGET}')
    else:
        pop_list = None
    # mask for conserved sites and repeats
    mask_file = os.path.join(outdir, 'mask.tsv')
    mask = env.Command([mask_file, mask_file + '.tbi'],
                       [phastcons, repeats],
                       'python region_mask.py ${SOURCES} | sort -k1V -k2n -k3n'
                       ' | bgzip > ${TARGETS[0]} && tabix -s 1 -b 2 -e 3 '
                       '${TARGETS[0]}')
    # loop over chromosome bcf files
    chr_sites = {}
    ref_snps = {}
    chr_ancestral = {}
    chr_majorminor = {}
    chrs = [str(chr_int) for chr_int in range(1, 23)]
    for chr in chrs:
        # NOTE assumes chr is indicated by second '.'-delimited field in bcf
        # file name
        ref_bcf = glob.glob(ref_bcf_dir + '/*chr' + chr + '.*.bcf')
        assert len(ref_bcf) == 1
        ref_bcf = ref_bcf[0]
        tgt = os.path.join(outdir, 'refpop.snps.{}.tsv'.format(chr))
        src = [mask[0], ref_bcf]
        if pop_list is not None:
            src.append(pop_list)
        cmd = 'bcftools view -T ^${SOURCES[0]} '
        if pop_list is not None:
            cmd += '-S ${SOURCES[2]} '
        cmd += ' -m2 -M2 -v snps -c 1 -Ou '
        if debug:
            cmd += f' -r {chr}{debug_range}'
        cmd += ' ${SOURCES[1]} | '
        cmd += 'bcftools query -f '
        cmd += '\'%CHROM\\t%POS\\t%REF\\t%ALT\\t%AC\\t%AN\\n\' > ${TARGET}'
        ref_snps[chr] = env.Command(tgt, src, cmd)

        tgt = os.path.join(outdir, 'filtered_ref.{}.fa'.format(chr))
        cmd = f'samtools faidx ${{SOURCE}} {chr}'
        if debug:
            cmd += debug_range
        cmd += ' > ${TARGET}'
        filtered_ref = env.Command(tgt, hg19, cmd)

        tgt = os.path.join(outdir, 'filtered_anc.{}.fa'.format(chr))
        cmd = f'samtools faidx ${{SOURCE}} {chr}'
        if debug:
            cmd += debug_range
        cmd += ' > ${TARGET}'
        filtered_anc = env.Command(tgt, hg19_ancestral, cmd)

        tgt = os.path.join(outdir, 'ancestral.{}.fa'.format(chr))
        src = [filtered_ref, filtered_anc + ref_snps[chr]]
        cmd = 'python ancestral_states.py ${SOURCES} > ${TARGET}'
        chr_ancestral[chr] = env.Command(tgt, src, cmd)
