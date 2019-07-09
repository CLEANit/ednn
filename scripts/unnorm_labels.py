#!/usr/bin/env python

import h5py
import numpy as np
import sys

h5file = h5py.File(sys.argv[1])

min_e = h5file.attrs['min']
max_e = h5file.attrs['max']
print 'Attributes min and max energies:', min_e, max_e
print 'Min and max before conversion:', np.min(h5file['Y']), np.max(h5file['Y'])
h5file['Y'][:] *= (max_e - min_e)
h5file['Y'][:] += min_e
print 'Min and max after conversion:', np.min(h5file['Y']), np.max(h5file['Y'])

