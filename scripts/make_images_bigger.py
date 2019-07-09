#!/usr/bin/env python

import numpy as np
import h5py
import sys

f = h5py.File(sys.argv[1], 'r')
new_f = h5py.File('bigger_images.h5', 'w')

factor = 3
new_f.create_dataset('X', (len(f['Y']),) +  tuple(np.array(f['X'].shape[1:-1]) * factor) + (1,))
new_f.create_dataset('Y', (len(f['Y']), 1))

power = len(f['X'].shape[1:-1])
new_f['Y'][:] = f['Y'][:] 
for key, val in f.attrs.items():
	new_f.attrs[key] = val

for i, X in enumerate(f['X']):
	new_f['X'][i] = np.tile(X, tuple([factor] * power) + (1,))
f.close()
new_f.close()


