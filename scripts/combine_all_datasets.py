#!/usr/bin/env python

import os
import h5py
import numpy as np
import sys

files = os.listdir(sys.argv[1])

total = h5py.File('total.h5', 'w')

total_len = 0

h5_files = []
for i, file in enumerate(files):
    h5_files.append(h5py.File(sys.argv[1] + file, 'r'))


for f in h5_files:
    total_len += len(f['Y'])
total.create_dataset('Y', (total_len,) + h5_files[0]['Y'].shape[1:])
total.create_dataset('X', (total_len,) + h5_files[0]['X'].shape[1:])

counter = 0
minval = np.min(h5_files[0]['Y'])
maxval = np.max(h5_files[0]['Y'])

for f in h5_files:
    for x,y in zip(f['X'], f['Y']):
        total['X'][counter] = x
        total['Y'][counter] = y
        counter += 1
        sys.stdout.write('Progress: %2.2f\r' % (100. * counter / total_len))
    if np.min(f['Y']) < minval:
        minval = np.min(f['Y'])
    if np.max(f['Y']) > maxval:
        maxval = np.max(f['Y'])

	f.close()
total.attrs['min'] = minval
total.attrs['max'] = maxval

total.close()

