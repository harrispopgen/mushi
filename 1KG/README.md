1000 Genomes Pipeline
==

Using NYGC 30x call set

Set up environment
--
```bash
conda env create -f env.yml
conda activate 1KG
```

SCons Pipeline
--
To print usage:
```bash
scons -h
```
(note "Local Options")

Example SGE job script: [`qsub_scons.sh`](qsub_scons.sh)
