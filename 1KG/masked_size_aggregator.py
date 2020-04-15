#! /usr/bin/env python

import sys
from collections import Counter

sizes = Counter()
for f in sys.argv[1:]:
    for line in open(f):
        context, count = line.split()
        sizes[context] += int(count)

for context in sorted(sizes):
    print(f'{context}\t{sizes[context]}')
