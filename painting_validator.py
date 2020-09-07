#!/usr/bin/env python 
import sys
import re

file_1 = sys.argv[1]
file_2 = sys.argv[2]

seeds_1 = re.findall("\d{16}, matches: \d{1,2}.", open(file_1).read())
seeds_2 = re.findall("\d{16}, matches: \d{1,2}.", open(file_2).read())

seeds_1 = sorted(list(set(seeds_1)))
seeds_2 = sorted(list(set(seeds_2)))

if seeds_1 == seeds_2:
    sys.exit(0)
else:
    sys.exit(1)